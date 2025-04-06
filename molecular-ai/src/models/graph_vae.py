# molecular-ai/src/models/graph_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import logging
import math
from typing import List, Optional, Tuple # Added types

# Add RDKit imports conditionally to avoid hard dependency if not installed
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

logger = logging.getLogger(__name__)

# --- LoRA Layer Definition ---
class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Check if the layer has bias
        self.has_bias = linear_layer.bias is not None

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Scaling factor
        self.scaling = self.lora_alpha / self.r

        # Freeze original weights and bias
        self.linear.weight.requires_grad = False
        if self.has_bias:
            self.linear.bias.requires_grad = False

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear layer output
        result = self.linear(x)
        
        # LoRA adaptation
        lora_update = self.lora_B @ self.lora_A # Calculate BA
        lora_update = lora_update.to(x.dtype) # Ensure dtype match
        
        # Apply LoRA update, potentially needing broadcast if x has batch dim
        # Adjusting matmul for potential batch dim in x
        if x.dim() == 2:
             # Standard (batch_size, in_features) @ (in_features, out_features)
             lora_result = (self.lora_dropout(x) @ lora_update.T) * self.scaling
        elif x.dim() == 3:
             # Potential (batch_size, seq_len, in_features)
             # We expect (batch_size, in_features) or (in_features)
             # This case might need specific handling depending on usage
             # Assuming simple linear layer usage on last dim: (..., in_features)
             lora_result = (self.lora_dropout(x) @ lora_update.T) * self.scaling
        else:
             # Handle other dimensions or raise error
             raise ValueError(f"Input tensor x has unexpected dimensions: {x.dim()}")

        result += lora_result
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        # Ensure original weights remain frozen during training
        self.linear.weight.requires_grad = False
        if self.has_bias:
            self.linear.bias.requires_grad = False

# --- MPNN Encoder with LoRA --- 
class MPNNEncoderVAE(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, latent_dim: int, 
                 num_layers: int, heads: int, dropout_p: float, 
                 lora_r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout_p))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout_p))
            
        # --- LoRA applied to output layers --- 
        # Create standard linear layers first
        fc_mu_base = nn.Linear(hidden_dim, latent_dim)
        fc_log_var_base = nn.Linear(hidden_dim, latent_dim)
        
        # Wrap them with LoRA
        self.fc_mu = LoRALinear(fc_mu_base, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.fc_log_var = LoRALinear(fc_log_var_base, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x, edge_index, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None or x.shape[0] == 0:
            num_graphs = torch.max(batch).item() + 1 if batch is not None else 1
            zero_latent = torch.zeros(num_graphs, self.latent_dim, device=edge_index.device if edge_index is not None else x.device)
            return zero_latent, zero_latent # mu, log_var
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        x = self.node_encoder(x)
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.relu(x)
            
        graph_embedding = global_mean_pool(x, batch)
        mu = self.fc_mu(graph_embedding)
        log_var = self.fc_log_var(graph_embedding)
        return mu, log_var

# --- MPNN Decoder with LoRA --- 
class MPNNDecoderVAE(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, node_features: int, 
                 max_nodes: int, num_dec_layers: int, heads: int, dropout_p: float,
                 lora_r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Initial MLP to project z to initial node states
        self.z_to_nodes = nn.Sequential(
            LoRALinear(nn.Linear(latent_dim, hidden_dim * max_nodes), r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.ReLU()
        )
        
        # Decoder GAT layers for message passing
        self.gat_layers = nn.ModuleList()
        gat_input_dim = hidden_dim + latent_dim 
        for _ in range(num_dec_layers):
             # Input dim is now hidden_dim + latent_dim, output is hidden_dim
            self.gat_layers.append(GATConv(gat_input_dim, hidden_dim // heads, heads=heads, dropout=dropout_p))
            # Update input dim for subsequent layers if needed (simplest: keep it hidden_dim)
            # If output of GATConv is just hidden_dim, next layer needs hidden_dim + latent_dim again
            # Let's assume GATConv outputs hidden_dim and we re-concat z each time
            # No, GATConv's input dim is fixed once defined. Let's make GAT output hidden_dim
            # And we'll concat z *before* each layer call in forward.

        # Predict node features from final node embeddings
        node_predictor_base = nn.Linear(hidden_dim, node_features)
        self.node_predictor = LoRALinear(node_predictor_base, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        # Predict adjacency logits from pairs of final node embeddings
        adj_predictor_base = nn.Linear(hidden_dim * 2, 1)
        self.adj_predictor = LoRALinear(adj_predictor_base, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # --- Add Bond Type Predictor Head ---
        self.num_bond_types = 4 # SINGLE, DOUBLE, TRIPLE, AROMATIC
        bond_type_predictor_base = nn.Linear(hidden_dim * 2, self.num_bond_types)
        self.bond_type_predictor = LoRALinear(bond_type_predictor_base, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # ------------------------------------

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Return 3 tensors now
        batch_size = z.size(0)
        device = z.device
        
        # 1. Project z to initial node states [batch_size * max_nodes, hidden_dim]
        node_states_flat = self.z_to_nodes(z)
        node_states = node_states_flat.view(batch_size, self.max_nodes, self.hidden_dim)
        
        # 2. Decoder Message Passing (on assumed fully connected graph within each sample)
        # Create dense edge_index for a fully connected graph of size max_nodes
        # Note: This is computationally expensive for large max_nodes!
        adj_full = torch.ones(self.max_nodes, self.max_nodes, device=device)
        adj_full.fill_diagonal_(0) # No self-loops initially
        edge_index_full, _ = dense_to_sparse(adj_full)
        edge_index_full = edge_index_full.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, 2, -1)
        
        # Reshape node_states for batch processing with GATConv
        node_states_gat = node_states.view(-1, self.hidden_dim) # Shape [batch*max_nodes, hidden]
        
        # Create batch vector for the fully connected graphs
        batch_vector = torch.arange(batch_size, device=device).repeat_interleave(self.max_nodes)
        
        # Adapt edge_index for the batch
        edge_indices_list = []
        for i in range(batch_size):
            edge_indices_list.append(edge_index_full[i] + i * self.max_nodes)
        edge_index_batched = torch.cat(edge_indices_list, dim=1)

        x = node_states_gat
        # --- Modification: Replicate z for concatenation ---
        # z has shape [batch_size, latent_dim]
        # Repeat z for each node within the batch
        z_rep = z.repeat_interleave(self.max_nodes, dim=0) # Shape [batch*max_nodes, latent_dim]
        # ------------------------------------------------

        for gat_layer in self.gat_layers:
            # GATConv expects [num_nodes, features], edge_index [2, num_edges]
            # --- Modification: Concatenate z with node features before passing to GAT --- 
            x_concat = torch.cat([x, z_rep], dim=1) # Shape [batch*max_nodes, hidden_dim + latent_dim]
            x = gat_layer(x_concat, edge_index_batched)
            # -------------------------------------------------------------------------
            x = F.relu(x)
            
        # Reshape back to [batch_size, max_nodes, hidden_dim]
        final_node_states = x.view(batch_size, self.max_nodes, self.hidden_dim)

        # 3. Predict Node Features
        recon_nodes = self.node_predictor(final_node_states)
        
        # 4. Predict Adjacency Logits & Bond Type Logits
        adj_logits = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=device)
        bond_type_logits = torch.zeros(batch_size, self.max_nodes, self.max_nodes, self.num_bond_types, device=device)
        for i in range(self.max_nodes):
            for j in range(i + 1, self.max_nodes): # Avoid self-loops and duplicates
                node_i_state = final_node_states[:, i, :] # [batch, hidden]
                node_j_state = final_node_states[:, j, :] # [batch, hidden]
                pair_features = torch.cat([node_i_state, node_j_state], dim=1) # [batch, hidden * 2]
                
                # Predict Adjacency
                edge_logit = self.adj_predictor(pair_features).squeeze(-1) # [batch]
                adj_logits[:, i, j] = edge_logit
                adj_logits[:, j, i] = edge_logit # Ensure symmetry
                
                # Predict Bond Type Logits
                bond_logits_pair = self.bond_type_predictor(pair_features) # [batch, num_bond_types]
                bond_type_logits[:, i, j, :] = bond_logits_pair
                bond_type_logits[:, j, i, :] = bond_logits_pair # Ensure symmetry
                
        return recon_nodes, adj_logits, bond_type_logits # Return bond types

# --- Updated GraphVAE --- 
class GraphVAE(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, latent_dim: int, 
                 max_nodes: int, num_enc_layers: int, heads_enc: int, dropout_enc: float, 
                 num_dec_layers: int, heads_dec: int, dropout_dec: float, 
                 lora_r: int, lora_alpha: float, lora_dropout: float):
        """Initialize GraphVAE with MPNN Encoder and MPNN Decoder + LoRA layers.

        Args:
            node_features: Dimensionality of node features.
            hidden_dim: Dimensionality of hidden layers in GCN and MLP.
            latent_dim: Dimensionality of the latent space.
            max_nodes: Maximum number of nodes expected in a graph (for decoder).
                       Note: This imposes a fixed-size constraint for the simple decoder.
            num_enc_layers: Number of layers in the MPNN encoder.
            heads_enc: Number of attention heads in the GAT layers for encoder.
            dropout_enc: Dropout probability for the GAT layers for encoder.
            num_dec_layers: Number of layers in the MPNN decoder.
            heads_dec: Number of attention heads in the GAT layers for decoder.
            dropout_dec: Dropout probability for the GAT layers for decoder.
            lora_r: Rank of the LoRA matrices.
            lora_alpha: Scaling factor for the LoRA matrices.
            lora_dropout: Dropout probability for the LoRA matrices.
        """
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # === Encoder ===
        self.encoder = MPNNEncoderVAE(
            node_features=node_features,
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim,
            num_layers=num_enc_layers,
            heads=heads_enc,
            dropout_p=dropout_enc,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # === Decoder ===
        self.decoder = MPNNDecoderVAE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            node_features=node_features,
            max_nodes=max_nodes,
            num_dec_layers=num_dec_layers,
            heads=heads_dec,
            dropout_p=dropout_dec,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Store atom map (example: needs proper definition based on dataset preprocessing)
        self.atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17} # Map index to atomic number
        self.bond_decoder_m = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE, 3: Chem.BondType.AROMATIC}

    def reparameterize(self, mu, log_var):
        """Applies the reparameterization trick to sample latent vector z."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Sample from standard normal distribution
        return mu + eps * std

    def forward(self, data):
        """Full forward pass: encode -> reparameterize -> decode."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mu, log_var = self.encoder(x, edge_index, batch)
        z = self.reparameterize(mu, log_var)
        recon_nodes, recon_adj_logits, recon_bond_logits = self.decoder(z) # Get bond logits
        return recon_nodes, recon_adj_logits, recon_bond_logits, mu, log_var # Return bond logits

    @staticmethod
    def loss_function(recon_nodes, recon_adj_logits, recon_bond_logits, data, mu, log_var, beta=1.0, kld_free_bits_lambda=0.0):
        """Calculates the VAE loss, now including bond type reconstruction."""
        batch_size = data.num_graphs
        device = recon_nodes.device
        max_nodes = recon_nodes.size(1) # Get max_nodes from the reconstructed tensor
        node_features_dim = recon_nodes.size(2)

        # --- Reconstruction Loss ---
        total_node_recon_loss = 0.0
        start_node_idx = 0
        for i in range(batch_size):
            num_nodes_in_graph = torch.sum(data.batch == i).item()
            if num_nodes_in_graph == 0:
                continue # Skip if graph is empty
                
            # Clamp to max_nodes if original graph was larger
            num_nodes_to_compare = min(num_nodes_in_graph, max_nodes)
            
            # Get original node features for this graph
            end_node_idx = start_node_idx + num_nodes_in_graph
            target_nodes_graph = data.x[start_node_idx:end_node_idx]
            
            # Get reconstructed features for this graph (up to num_nodes_to_compare)
            recon_nodes_graph = recon_nodes[i, :num_nodes_to_compare, :]
            
            # Calculate MSE loss for the nodes present in both original (up to max_nodes) and reconstruction
            # Ensure target tensor slice matches the number of nodes we compare
            loss_for_graph = F.mse_loss(
                recon_nodes_graph, 
                target_nodes_graph[:num_nodes_to_compare, :], 
                reduction='mean'
            )
            total_node_recon_loss += loss_for_graph
            
            # Update starting index for the next graph
            start_node_idx = end_node_idx

        node_recon_loss = total_node_recon_loss / batch_size # Average over batch size

        # 2. Adjacency Reconstruction (BCE With Logits)
        # Ensure target_adj is created with the same max_num_nodes as recon_adj_logits
        target_adj = to_dense_adj(data.edge_index, batch=data.batch, max_num_nodes=max_nodes).float()
        adj_recon_loss = F.binary_cross_entropy_with_logits(
            recon_adj_logits, target_adj, reduction='mean'
        )

        # 3. Bond Type Reconstruction (Cross Entropy)
        num_bond_types = recon_bond_logits.shape[-1]
        bond_type_target = torch.zeros_like(recon_bond_logits) # Shape [B, N, N, C]
        
        if data.edge_index.numel() > 0 and data.edge_attr is not None:
            bond_indices = data.edge_index.t() # Shape [num_edges, 2]
            batch_edge_indices = data.batch[bond_indices[:, 0]] # Get batch index for each edge's source node

            # Get the one-hot bond types from edge_attr
            true_bond_types_one_hot = data.edge_attr[:, :num_bond_types] # Shape [num_edges, C]

            start_node_idx = 0
            for i_graph in range(batch_size):
                num_nodes_in_graph = torch.sum(data.batch == i_graph).item()
                if num_nodes_in_graph == 0: continue

                edges_in_graph_mask = (batch_edge_indices == i_graph)
                graph_edge_indices = bond_indices[edges_in_graph_mask] # Edges for this graph
                graph_edge_attr = true_bond_types_one_hot[edges_in_graph_mask] # Bond types for this graph

                # Map global node indices to local 0..N-1 indices for this graph
                local_edge_indices = graph_edge_indices - start_node_idx

                # Populate the dense target matrix for this graph
                # Ensure indices are within [0, max_nodes-1]
                valid_edge_mask = (local_edge_indices[:, 0] < max_nodes) & (local_edge_indices[:, 1] < max_nodes)
                valid_local_indices = local_edge_indices[valid_edge_mask]
                valid_edge_attr = graph_edge_attr[valid_edge_mask]
                
                if valid_local_indices.numel() > 0:
                    # Use advanced indexing to populate bond_type_target
                    batch_indices_for_edges = torch.full_like(valid_local_indices[:, 0], i_graph)
                    # Convert one-hot target to class indices for cross_entropy
                    # Note: Need target indices, not one-hot for F.cross_entropy later
                    # Store one-hot for now, convert when calculating loss
                    bond_type_target[batch_indices_for_edges, valid_local_indices[:, 0], valid_local_indices[:, 1], :] = valid_edge_attr

                start_node_idx += num_nodes_in_graph

        # Mask the loss calculation to only existing edges in the target graph
        mask = target_adj.unsqueeze(-1).expand_as(recon_bond_logits) > 0 # Expand mask to bond type dim
        num_masked_elements = mask.sum().item()

        if num_masked_elements > 0:
            recon_bond_logits_masked = recon_bond_logits[mask].view(-1, num_bond_types)
            target_bond_types_masked_indices = bond_type_target[mask].view(-1, num_bond_types).argmax(dim=1)

            bond_type_loss = F.cross_entropy(
                recon_bond_logits_masked,
                target_bond_types_masked_indices, 
                reduction='mean'
            )
        else:
            bond_type_loss = torch.tensor(0.0, device=device) # No edges, no loss

        # Add bond type loss to overall reconstruction loss
        recon_loss = node_recon_loss + adj_recon_loss + bond_type_loss

        # --- KL Divergence ---
        # --- Modification: Apply Free Bits --- 
        # kld_free_bits_lambda = 0.05 # Removed hardcoded default, now passed in
        # Calculate KLD per latent dimension: 0.5 * sum(1 + log_var - mu^2 - sigma^2) for each dim
        kld_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        
        # Apply free bits threshold per dimension if lambda > 0
        if kld_free_bits_lambda > 0:
            free_bits_kld = torch.clamp(kld_per_dim - kld_free_bits_lambda, min=0.).sum() / batch_size
            kld_term_for_loss = free_bits_kld
        else:
            # If lambda is 0 or less, use original KLD calculation
            original_kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
            kld_term_for_loss = original_kld_loss
        # --- End Modification ---
        
        # --- Original KLD Calculation (for logging) ---
        original_kld_loss_log = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

        # --- Total Loss (Using Free Bits KLD if applicable) ---
        total_loss = recon_loss + beta * kld_term_for_loss

        return total_loss, {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "node_recon_loss": node_recon_loss.item(),
            "adj_recon_loss": adj_recon_loss.item(),
            "bond_type_loss": bond_type_loss.item(),
            # Log both the KLD used in loss and the original KLD for monitoring
            "kld_loss": kld_term_for_loss.item(), 
            "original_kld_loss": original_kld_loss_log.item()
        }

    # --- NEW sample_smiles method ---
    def sample_smiles(self, z: torch.Tensor, adj_threshold: float = 0.5) -> List[Optional[str]]:
        """Samples from the latent space, decodes, and converts to SMILES."""
        if not RDKIT_AVAILABLE:
            logger.error("RDKit is not available. Cannot generate SMILES.")
            return [None] * z.size(0)
            
        self.decoder.eval() # Ensure decoder is in eval mode
        with torch.no_grad():
            recon_nodes, adj_logits, bond_type_logits = self.decoder(z)
            adj_prob = torch.sigmoid(adj_logits) # Probabilities for adjacency
            node_types_indices = torch.argmax(recon_nodes, dim=-1) # Get node type index
            bond_types_indices = torch.argmax(bond_type_logits, dim=-1) # Get bond type index
            
        smiles_list = []
        batch_size = z.size(0)
        
        for i in range(batch_size):
            mol = None
            try:
                adj_matrix_i = (adj_prob[i] > adj_threshold).int() # Threshold adjacency
                node_indices_i = node_types_indices[i]
                bond_indices_i = bond_types_indices[i]
                
                # Assume first N valid nodes based on some criteria or just use all max_nodes?
                # For simplicity, let's try building with all max_nodes first, RDKit might handle extra nodes.
                # A better approach might be to predict the number of nodes or use a stop token.
                num_nodes = self.max_nodes 
                
                rw_mol = Chem.RWMol() # Editable molecule
                
                # Add atoms
                actual_node_indices = []
                for node_idx in range(num_nodes):
                    atom_type_idx = node_indices_i[node_idx].item()
                    if atom_type_idx in self.atom_decoder_m:
                        atomic_num = self.atom_decoder_m[atom_type_idx]
                        atom = Chem.Atom(atomic_num)
                        rw_mol.AddAtom(atom)
                        actual_node_indices.append(node_idx) # Keep track of added atoms
                    else:
                         # Handle unknown atom type index if necessary
                         logger.warning(f"Sample {i}: Unknown atom type index {atom_type_idx} at node {node_idx}")
                         # Add a placeholder like Carbon? Or skip?
                         # rw_mol.AddAtom(Chem.Atom(6)) # Example: Add Carbon 
                         pass # Skip adding atom for now

                # Adjust number of nodes based on successfully added atoms
                num_valid_nodes = rw_mol.GetNumAtoms()
                if num_valid_nodes == 0:
                    smiles_list.append(None)
                    continue
                
                # Add bonds
                for r in range(num_valid_nodes):
                    for c in range(r + 1, num_valid_nodes):
                        # Map back from rw_mol index (r, c) to original node index in max_nodes
                        orig_r = actual_node_indices[r]
                        orig_c = actual_node_indices[c]
                        
                        if adj_matrix_i[orig_r, orig_c].item() == 1:
                            bond_type_idx = bond_indices_i[orig_r, orig_c].item()
                            if bond_type_idx in self.bond_decoder_m:
                                bond_type = self.bond_decoder_m[bond_type_idx]
                                rw_mol.AddBond(r, c, bond_type)
                            else:
                                logger.warning(f"Sample {i}: Unknown bond type index {bond_type_idx} between nodes {orig_r}-{orig_c}")
                                # Add a single bond as default?
                                # rw_mol.AddBond(r, c, Chem.BondType.SINGLE)
                                pass # Skip adding bond for now
                
                # Finalize molecule
                mol = rw_mol.GetMol()
                Chem.SanitizeMol(mol) # Check valencies and aromaticity
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
                
            except Exception as e:
                # Log error if RDKit fails (e.g., sanitization error)
                # logger.debug(f"Sample {i}: RDKit error during SMILES generation: {e}")
                smiles_list.append(None) # Append None if generation failed
                
        return smiles_list
    # --- End NEW sample_smiles method ---

    # Keep the old sample method for compatibility if needed, but mark as deprecated? 
    # Or remove it if sample_smiles is the intended replacement.
    # @staticmethod 
    # def sample(decoder, num_samples: int, latent_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Samples from the latent space and decodes."""
    #     with torch.no_grad():
    #         z = torch.randn(num_samples, latent_dim).to(device)
    #         node_features_pred, adj_logits_pred, _ = decoder(z) # Ignore bond types here
    #         adj_prob = torch.sigmoid(adj_logits_pred)
    #     return node_features_pred, adj_prob