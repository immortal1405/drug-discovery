# molecular-ai/src/rl/optimize_gcpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import logging
import os
import copy
from typing import List, Dict, Any, Tuple, Optional

# Assume VAE model is in src.models.graph_vae
from ..models.graph_vae import GraphVAE
# Assume reward calculation utils are available
from ..utils.rewards import calculate_reward # type: ignore # Keep this import
# Placeholder for property predictor loading
from ..utils.predictors import load_property_predictors # type: ignore # Keep this import


# RDKit for molecule manipulation during refinement
# ... (RDKit import remains the same) ...
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem # For adding atoms/bonds
    RDKIT_AVAILABLE = True
    RDLogger.DisableLog('rdApp.*') # Disable RDKit excessive logging
except ImportError:
    RDKIT_AVAILABLE = False
    logging.error("RDKit is required for GCPN refinement steps!")
    exit() # Cannot proceed without RDKit

# PyTorch Geometric for GNNs
# ... (PyG import remains the same) ...
try:
    import torch_geometric.nn as gnn
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_networkx, from_networkx # For potential conversions
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.error("PyTorch Geometric (torch_geometric) is required for GCPN!")
    exit() # Cannot proceed without PyG

# Setup logging
# ... (logging setup remains the same) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Mappings (Ensure these match VAE preprocessing) ---
# ... (Constants and Mappings remain the same) ...
ATOM_MAP = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8} # Example: C:0, N:1, O:2...
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53] # Atomic numbers corresponding to ATOM_MAP indices
ATOM_FEATURE_DIM = len(ATOM_MAP)
BOND_MAP = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
BOND_FEATURE_DIM = len(BOND_MAP)

# --- Refined Action Space ---
# ... (Action Space remains the same) ...
ACTION_TYPES = [
    "STOP",
    "ADD_ATOM",     # Add ATOM_TYPE, connect to TARGET_NODE with SINGLE bond
    "CHANGE_ATOM",  # Change TARGET_NODE to ATOM_TYPE
    "REMOVE_ATOM",  # Remove TARGET_NODE
]
NUM_ACTION_TYPES = len(ACTION_TYPES)

# --- Helper Function: Convert RDKit Mol to PyG Data ---
# ... (mol_to_pyg_data function remains the same) ...
def mol_to_pyg_data(mol: Chem.Mol, device: torch.device = torch.device('cpu')) -> Optional[Data]:
    """ Converts an RDKit Mol object to a PyTorch Geometric Data object. """
    if mol is None: return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False) # Work without Hs for simplicity
        # Node features (atom types - one-hot encoded)
        atom_features = []
        valid_indices = []
        node_map = {} # Map old index to new index (0..N-1)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_type_idx = ATOM_MAP.get(atom.GetAtomicNum())
            if atom_type_idx is not None: # Only include atoms in our map
                feature = [0.0] * ATOM_FEATURE_DIM
                feature[atom_type_idx] = 1.0
                atom_features.append(feature)
                node_map[i] = len(valid_indices)
                valid_indices.append(i)

        if not atom_features: return None # No valid atoms found
        x = torch.tensor(atom_features, dtype=torch.float, device=device)

        # Edge index and features
        rows, cols, edge_features = [], [], []
        if mol.GetNumBonds() > 0:
            for bond in mol.GetBonds():
                start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # Check if both atoms are included
                if start_idx in node_map and end_idx in node_map:
                    start_new, end_new = node_map[start_idx], node_map[end_idx]
                    bond_type_idx = BOND_MAP.get(bond.GetBondType())
                    if bond_type_idx is not None: # Check if bond type is valid
                        feature = [0.0] * BOND_FEATURE_DIM
                        feature[bond_type_idx] = 1.0
                        # Add edges in both directions
                        rows.extend([start_new, end_new])
                        cols.extend([end_new, start_new])
                        edge_features.extend([feature, feature])

            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
            edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float, device=device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = x.size(0)
        data.original_indices = torch.tensor(valid_indices, dtype=torch.long, device=device) # Store original RDKit indices
        return data

    except Exception as e:
        logger.error(f"Error converting Mol to PyG Data: {e}")
        return None

# --- GCPN Actor Network (Policy GNN) ---
# ... (ActorGNN class remains the same) ...
class ActorGNN(nn.Module):
    def __init__(self, node_in_features: int, edge_in_features: int, hidden_dim: int,
                 num_atom_types: int, num_action_types: int, num_layers: int = 3):
        super().__init__()
        self.node_emb = nn.Linear(node_in_features, hidden_dim)
        self.convs = nn.ModuleList([gnn.GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Heads for predicting different action components from the global graph embedding
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.target_node_head = nn.Linear(hidden_dim, 1) # Will use attention/softmax over node embeddings later
        self.new_atom_type_head = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, data: Data) -> Tuple[Categorical, Categorical, Categorical, torch.Tensor]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_emb(x).relu()
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # Global graph representation (using mean pooling)
        graph_embedding = gnn.global_mean_pool(x, batch) # [batch_size, hidden_dim]

        # 1. Predict Action Type
        action_type_logits = self.action_type_head(graph_embedding)
        action_type_dist = Categorical(logits=action_type_logits)

        # 2. Predict Target Node
        # Simple approach: Use graph embedding to predict.
        # Better approach: Use attention mechanism based on node embeddings 'x'
        # For simplicity now: Use node embeddings directly for logits
        target_node_logits = self.target_node_head(x).squeeze(-1) # [num_nodes_in_batch, 1] -> [num_nodes_in_batch]
        # Apply masking later based on batch
        target_node_dist = Categorical(logits=target_node_logits) # Will need per-graph sampling later

        # 3. Predict New Atom Type (for ADD_ATOM, CHANGE_ATOM)
        new_atom_type_logits = self.new_atom_type_head(graph_embedding)
        new_atom_type_dist = Categorical(logits=new_atom_type_logits)

        return action_type_dist, target_node_dist, new_atom_type_dist, x # Return node embeddings for critic


# --- GCPN Critic Network (Value GNN) ---
# ... (CriticGNN class remains the same) ...
class CriticGNN(nn.Module):
     def __init__(self, node_in_features: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        # Shares the GNN body structure with Actor is common practice
        self.node_emb = nn.Linear(node_in_features, hidden_dim)
        self.convs = nn.ModuleList([gnn.GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.value_head = nn.Linear(hidden_dim, 1)

     def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_emb(x).relu()
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        graph_embedding = gnn.global_mean_pool(x, batch)
        value = self.value_head(graph_embedding)
        return value

# --- Function to Apply Action to RDKit Molecule ---
# ... (apply_action function remains the same) ...
def apply_action(
    mol: Chem.Mol,
    action_type_idx: int,
    target_node_idx: Optional[int], # Index within the *current* PyG graph (0..N-1)
    new_atom_type_idx: Optional[int],
    original_indices_map: torch.Tensor # Map from PyG index to RDKit index
    ) -> Optional[Chem.Mol]:
    """
    Applies a modification action to an RDKit molecule using EditableMol.

    Returns: Modified molecule or None if action is invalid/fails.
    """
    action_type = ACTION_TYPES[action_type_idx]
    if action_type == "STOP":
        return mol # Return the molecule unchanged

    if target_node_idx is None or target_node_idx >= len(original_indices_map):
         # logger.debug(f"Invalid target_node_idx: {target_node_idx} for map size {len(original_indices_map)}")
         return None # Invalid target node index for the current graph

    # Map PyG node index back to original RDKit atom index
    # Ensure target_node_idx is within bounds of original_indices_map
    if not (0 <= target_node_idx < len(original_indices_map)):
         # logger.debug(f"target_node_idx {target_node_idx} out of bounds for original_indices_map")
         return None
    rdkit_target_idx = int(original_indices_map[target_node_idx].item())

    # Ensure rdkit_target_idx is valid for the *current* molecule
    if not (0 <= rdkit_target_idx < mol.GetNumAtoms()):
        # logger.debug(f"Mapped rdkit_target_idx {rdkit_target_idx} is out of bounds for molecule with {mol.GetNumAtoms()} atoms.")
        return None


    try:
        # Create a copy to avoid modifying the original molecule in case of failure
        editable_mol = Chem.EditableMol(copy.deepcopy(mol))

        if action_type == "ADD_ATOM":
            if new_atom_type_idx is None or new_atom_type_idx >= len(ATOM_LIST): return None
            new_atom_atomic_num = ATOM_LIST[new_atom_type_idx]
            new_atom = Chem.Atom(new_atom_atomic_num)
            # Ensure the target atom actually exists in the editable mol
            if rdkit_target_idx >= editable_mol.GetMol().GetNumAtoms(): return None
            new_atom_idx = editable_mol.AddAtom(new_atom)
            # Add single bond to target atom
            editable_mol.AddBond(rdkit_target_idx, new_atom_idx, Chem.BondType.SINGLE)

        elif action_type == "CHANGE_ATOM":
            if new_atom_type_idx is None or new_atom_type_idx >= len(ATOM_LIST): return None
            new_atom_atomic_num = ATOM_LIST[new_atom_type_idx]
            # Ensure the target atom actually exists in the editable mol
            if rdkit_target_idx >= editable_mol.GetMol().GetNumAtoms(): return None
            # ReplaceAtom preserves connectivity
            editable_mol.ReplaceAtom(rdkit_target_idx, Chem.Atom(new_atom_atomic_num))


        elif action_type == "REMOVE_ATOM":
             # Ensure the target atom actually exists in the editable mol
            if rdkit_target_idx >= editable_mol.GetMol().GetNumAtoms(): return None
            # Simple removal - might lead to invalid intermediate states or disconnections
            editable_mol.RemoveAtom(rdkit_target_idx)

        else: # Should not happen
            return None

        # Get molecule back from editable version
        modified_mol_raw = editable_mol.GetMol()

        # Basic check if molecule is empty after modification
        if modified_mol_raw.GetNumAtoms() == 0:
            # logger.debug("Molecule became empty after action.")
            return None

        # Sanitize to check chemical validity
        try:
            status = Chem.SanitizeMol(modified_mol_raw, catchErrors=True)
            if status != Chem.SanitizeFlags.SANITIZE_NONE:
                 # logger.debug(f"Sanitization failed after action {action_type}: {status}")
                 return None # Sanitization failed
            # Return the sanitized molecule if successful
            return modified_mol_raw
        except Exception as e:
             # logger.debug(f"Exception during sanitization after {action_type}: {e}")
             return None # Exception during sanitization

    except Exception as e:
        # logger.error(f"Error applying action {action_type} (target: {rdkit_target_idx}): {e}")
        return None

# --- Function to Calculate GAE ---
# ... (calculate_gae function remains the same) ...
def calculate_gae(rewards: List[float], values: List[torch.Tensor], dones: List[bool], gamma: float, gae_lambda: float, device: torch.device) -> torch.Tensor:
    """Calculates Generalized Advantage Estimation."""
    advantages = []
    last_advantage = 0.0
    # Ensure values list is not empty and handle the terminal state value
    # Adjust index for values which includes the final state value
    num_steps = len(rewards)

    for i in reversed(range(num_steps)):
        # Value corresponds to state S_i, next_value corresponds to S_{i+1}
        value = values[i].item()
        next_value = values[i+1].item()
        done_mask = 1.0 - float(dones[i]) # 0 if done at step i, 1 otherwise

        # Calculate TD error: R_i + gamma * V(S_{i+1}) * (1 - done_i) - V(S_i)
        delta = rewards[i] + gamma * next_value * done_mask - value
        # Calculate advantage using GAE formula
        last_advantage = delta + gamma * gae_lambda * done_mask * last_advantage
        advantages.append(last_advantage)

    advantages.reverse()
    # Detach advantages as they are used as targets
    return torch.tensor(advantages, dtype=torch.float32, device=device).detach()


# --- Main GCPN Optimization Function (Implemented TODOs) ---
def optimize_gcpn(
    vae_model_path: str,
    actor_save_path: str,
    critic_save_path: str,
    property_predictor_paths: Dict[str, str],
    # --- VAE Params ---
    # ** CRITICAL: VERIFY/UPDATE THESE TO MATCH YOUR TRAINED VAE MODEL **
    # ** Values below are updated based on molecular-ai/config/vae_config.json **
    # ** BUT node_features discrepancy exists (config=120, ATOM_MAP=9). Using ATOM_MAP value for now.**
    vae_latent_dim: int = 128,            # UPDATED: Changed from 64 to 128 to match trained model dimensions
    vae_node_features: int = 120,         # UPDATED: Using value from config (120) instead of ATOM_MAP size
    vae_max_nodes: int = 100,             # UPDATED: Changed from 50 to 100 based on tensor size mismatch
    vae_hidden_dim: int = 256,            # UPDATED: Changed from 128 to 256 to match trained model dimensions
    vae_num_enc_layers: int = 3,          # From config
    vae_heads_enc: int = 4,               # Assumed from config 'heads'
    vae_dropout_enc: float = 0.1,         # Default, not in config
    vae_num_dec_layers: int = 3,          # Assumed same as enc_layers
    vae_heads_dec: int = 4,               # Assumed from config 'heads'
    vae_dropout_dec: float = 0.1,         # Default, not in config
    # Assuming no LoRA if not explicitly mentioned
    vae_lora_r: int = 8,                  # UPDATED: Set to 8 instead of 0 to avoid division by zero
    vae_lora_alpha: float = 1.0,
    vae_lora_dropout: float = 0.0,
    # --- GCPN Params ---
    gcpn_hidden_dim: int = 256,          # UPDATED: Changed from 128 to 256 to match VAE
    gcpn_num_layers: int = 3,
    num_refinement_steps: int = 5, # Max edits per molecule
    # --- RL Params (PPO) ---
    num_iterations: int = 60,            # REDUCED: From 500 to 60 for faster completion
    steps_per_iteration: int = 256,      # REDUCED: From 1024 to 256 for faster iterations
    batch_size: int = 32,                # REDUCED: From 64 to 32 for faster updates
    learning_rate_actor: float = 3e-4,
    learning_rate_critic: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 5,                 # REDUCED: From 10 to 5 for faster updates
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01, # Encourages exploration
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.015, # Optional KL divergence target for early stopping
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    logger.info(f"Starting GCPN optimization on device: {device}")
    logger.warning("CRITICAL: Ensure VAE parameters in `optimize_gcpn` match your trained model!")
    logger.warning("CRITICAL: Ensure ATOM_MAP/BOND_MAP match VAE preprocessing!")
    logger.warning("Using placeholder property predictors. Replace loading and reward calculation for real optimization.")

    # --- 1. Load Pre-trained VAE (Frozen) ---
    logger.info(f"Loading VAE model from: {vae_model_path}")
    # Using the parameters defined above - ** USER MUST VERIFY **
    vae_args = {
        'node_features': vae_node_features, 
        'hidden_dim': vae_hidden_dim, 
        'latent_dim': vae_latent_dim,
        'max_nodes': vae_max_nodes, 
        'num_enc_layers': vae_num_enc_layers, 
        'heads_enc': vae_heads_enc, 
        'dropout_enc': vae_dropout_enc,
        'num_dec_layers': vae_num_dec_layers, 
        'heads_dec': vae_heads_dec, 
        'dropout_dec': vae_dropout_dec,
    }
    
    # Only add LoRA params if r > 0 to avoid division by zero
    if vae_lora_r > 0:
        vae_args.update({
            'lora_r': vae_lora_r,
            'lora_alpha': vae_lora_alpha,
            'lora_dropout': vae_lora_dropout
        })
        logger.info(f"Using LoRA with r={vae_lora_r}, alpha={vae_lora_alpha}")
    else:
        logger.info("LoRA disabled (r=0)")
    
    try:
        logger.info(f"VAE args: {vae_args}")
        
        # Enable detailed error printing
        torch.set_printoptions(precision=10, threshold=10000)
        
        # Print model structure before initializing
        logger.info(f"Model initialization with args: {vae_args}")
        
        # Create model with error handling for division by zero
        try:
            vae = GraphVAE(**vae_args).to(device)
            logger.info("VAE model created successfully")
            
            # Set atom and bond decoders as attributes AFTER initialization (they're not constructor parameters)
            # Create a more robust atom decoder that handles a wider range of indices
            # The error suggests the trained model has atom type indices up to at least 50
            extended_atom_list = list(range(121))  # Create indices 0-120 to match vae_node_features=120
            
            # Map all indices to Carbon (6) by default
            atom_decoder_m = {i: 6 for i in range(121)}
            
            # Then set our known mappings
            for i, atom_num in enumerate(ATOM_LIST):
                atom_decoder_m[i] = atom_num
                
            # Also map some standard indices that might be used in the trained model
            # Common SMILES atoms:       C    N    O    F    P    S    Cl   Br   I
            for i, atom_num in zip([0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 15, 16, 17, 35, 53]):
                atom_decoder_m[i] = atom_num
                
            # Add specific mapping for index 50 (which caused the error)
            atom_decoder_m[50] = 6  # Default to Carbon
            
            bond_decoder_m = {i: bond_type for i, bond_type in enumerate(BOND_MAP.keys())}
            vae.atom_decoder_m = atom_decoder_m
            vae.bond_decoder_m = bond_decoder_m
            logger.info("Added extended atom and bond decoders as attributes")
            
        except ZeroDivisionError as zde:
            logger.error(f"Division by zero during VAE initialization: {zde}")
            logger.error("This could be due to parameter mismatch. Try different node_features value.")
            raise
        except Exception as e:
            logger.error(f"Error during VAE creation: {e}")
            raise
            
        # Load state dict with error handling
        try:
            if not os.path.exists(vae_model_path):
                logger.error(f"VAE model file not found: {vae_model_path}")
                return
                
            state_dict = torch.load(vae_model_path, map_location=device)
            logger.info(f"Loaded state dict with keys: {list(state_dict.keys())}")
            vae.load_state_dict(state_dict)
            logger.info("State dict loaded successfully")
            
            vae.eval()
            for param in vae.parameters():
                param.requires_grad = False
            logger.info("VAE model loaded and frozen successfully.")
            
            # Test VAE sampling to ensure it works
            logger.info("Testing VAE sampling...")
            z_test = torch.randn(1, vae_latent_dim, device=device)
            test_smiles = vae.sample_smiles(z_test)
            logger.info(f"Test sampling result: {test_smiles}")
            
        except Exception as e:
            logger.error(f"Failed to load VAE state dict: {e}")
            raise

    except Exception as e:
        logger.error(f"Failed to load VAE model: {e}")
        return

    # --- 2. Load Property Predictors (Placeholder Implementation) ---
    try:
        # Replace this with actual loading logic if models exist
        property_predictors = load_property_predictors(property_predictor_paths, device)
        logger.info(f"Loaded property predictors: {list(property_predictors.keys())}")
    except Exception as e:
        logger.error(f"Failed to load property predictors (using placeholders): {e}")
        property_predictors = {} # Fallback to empty dict

    # --- 3. Initialize Actor & Critic GNNs + Optimizers ---
    # ... (Initialization remains the same) ...
    actor = ActorGNN(node_in_features=ATOM_FEATURE_DIM, edge_in_features=BOND_FEATURE_DIM,
                     hidden_dim=gcpn_hidden_dim, num_atom_types=len(ATOM_LIST),
                     num_action_types=NUM_ACTION_TYPES, num_layers=gcpn_num_layers).to(device)
    critic = CriticGNN(node_in_features=ATOM_FEATURE_DIM, hidden_dim=gcpn_hidden_dim,
                       num_layers=gcpn_num_layers).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate_actor, eps=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate_critic, eps=1e-5)
    logger.info("Actor, Critic, and Optimizers initialized.")

    # --- 4. RL Training Loop (PPO) ---
    # ... (RL Loop structure remains the same, including buffer setup) ...
    global_step = 0
    current_mol = None
    current_step_in_traj = 0

    buffer_states = []
    buffer_actions_type = []
    buffer_actions_target = []
    buffer_actions_new_atom = []
    buffer_log_probs_type = []
    buffer_log_probs_target = []
    buffer_log_probs_new_atom = []
    buffer_rewards = []
    buffer_values = []
    buffer_dones = []
    buffer_original_indices = []

    for iteration in range(num_iterations):
        actor.eval()
        critic.eval()
        collected_steps = 0
        iteration_final_rewards = []
        iteration_final_smiles = [] # Track final SMILES per iteration

        # --- 4a. Collect Trajectories (Rollout Phase) ---
        logger.info(f"Collecting trajectories for iteration {iteration+1}/{num_iterations}")
        trajectory_count = 0
        successful_trajectories = 0
        
        while collected_steps < steps_per_iteration:
            # Reset or initialize state
            if current_mol is None or traj_done: # traj_done ensures reset after STOP/failure
                trajectory_count += 1
                z_initial = torch.randn(1, vae_latent_dim, device=device)
                try:
                    # Ensure VAE sampling doesn't crash
                    initial_smiles_list = vae.sample_smiles(z_initial)
                    initial_smiles = initial_smiles_list[0] if initial_smiles_list and initial_smiles_list[0] else None
                    if initial_smiles and '.' in initial_smiles:
                        logger.debug(f"Initial SMILES contains disconnected fragments: {initial_smiles}")
                        
                        # Try to take the first fragment instead
                        fragments = initial_smiles.split('.')
                        if len(fragments) > 0 and Chem.MolFromSmiles(fragments[0]):
                            initial_smiles = fragments[0]
                            logger.debug(f"Using first fragment: {initial_smiles}")
                except Exception as e:
                    logger.warning(f"VAE sampling failed: {e}")
                    initial_smiles = None

                if not initial_smiles: 
                    if trajectory_count % 20 == 0:
                        logger.info(f"Trajectory {trajectory_count}: VAE generated invalid SMILES, retrying...")
                    
                    # After several failed attempts, create a simple molecule
                    if trajectory_count % 100 == 0:
                        logger.info("Creating simple molecule after repeated VAE failures")
                        current_mol = create_sample_molecule()
                        if current_mol:
                            current_step_in_traj = 0
                            traj_done = False
                            simple_smiles = Chem.MolToSmiles(current_mol)
                            logger.info(f"Using simple molecule: {simple_smiles}")
                            continue
                            
                    continue # VAE failed

                current_mol = Chem.MolFromSmiles(initial_smiles)
                if not current_mol: 
                    current_mol = None
                    if trajectory_count % 20 == 0:
                        logger.info(f"Trajectory {trajectory_count}: Invalid SMILES from VAE: {initial_smiles}")
                    continue # Invalid initial SMILES

                current_step_in_traj = 0
                traj_done = False
                if trajectory_count % 20 == 0:
                    logger.info(f"Trajectory {trajectory_count}: Starting with SMILES {initial_smiles}")

            # --- Interact with environment ---
            state_data = mol_to_pyg_data(current_mol, device=device)
            if not state_data:
                traj_done = True
                final_reward = -1.0 # Penalize conversion failure heavily
                buffer_rewards.append(final_reward)
                buffer_dones.append(traj_done)
                buffer_states.append(buffer_states[-1].cpu() if buffer_states else None) # Append dummy state? Needs care
                buffer_original_indices.append(buffer_original_indices[-1].cpu() if buffer_original_indices else None)
                buffer_actions_type.append(torch.tensor(ACTION_TYPES.index("STOP"))) # Fake action
                buffer_actions_target.append(torch.tensor(0))
                buffer_actions_new_atom.append(torch.tensor(0))
                buffer_log_probs_type.append(torch.tensor(0.0))
                buffer_log_probs_target.append(torch.tensor(0.0))
                buffer_log_probs_new_atom.append(torch.tensor(0.0))
                buffer_values.append(torch.tensor([[0.0]])) # Zero value
                collected_steps += 1
                global_step += 1
                current_mol = None # Reset
                continue

            # Get action distributions and value
            with torch.no_grad():
                state_batch = Batch.from_data_list([state_data]).to(device)
                action_type_dist, target_node_dist_full, new_atom_type_dist, node_embeddings = actor(state_batch)
                value = critic(state_batch)

                # Sample target node specific to this graph
                node_indices_this_graph = (state_batch.batch == 0).nonzero(as_tuple=False).squeeze()
                target_node_logits_this_graph = target_node_dist_full.logits[node_indices_this_graph]
                if len(target_node_logits_this_graph.shape) == 0: # Handle single node case
                    target_node_logits_this_graph = target_node_logits_this_graph.unsqueeze(0)

                if target_node_logits_this_graph.nelement() == 0: # Graph has no valid nodes
                    action_type = torch.tensor(ACTION_TYPES.index("STOP"), device=device)
                    target_node = torch.tensor(0, device=device) # Dummy
                    new_atom_type = torch.tensor(0, device=device) # Dummy
                    log_prob_target = torch.tensor(0.0, device=device)
                else:
                    target_node_dist_this_graph = Categorical(logits=target_node_logits_this_graph)
                    action_type = action_type_dist.sample()
                    target_node = target_node_dist_this_graph.sample()
                    new_atom_type = new_atom_type_dist.sample()
                    log_prob_target = target_node_dist_this_graph.log_prob(target_node)

            # Store experience
            buffer_states.append(state_data.cpu())
            buffer_original_indices.append(state_data.original_indices.cpu())
            buffer_actions_type.append(action_type.cpu())
            buffer_actions_target.append(target_node.cpu())
            buffer_actions_new_atom.append(new_atom_type.cpu())
            log_prob_type = action_type_dist.log_prob(action_type)
            log_prob_new_atom = new_atom_type_dist.log_prob(new_atom_type)
            buffer_log_probs_type.append(log_prob_type.cpu())
            buffer_log_probs_target.append(log_prob_target.cpu())
            buffer_log_probs_new_atom.append(log_prob_new_atom.cpu())
            buffer_values.append(value.cpu())

            # Apply action
            action_type_idx = action_type.item()
            target_node_idx = target_node.item() if target_node_logits_this_graph.nelement() > 0 else None
            new_atom_type_idx = new_atom_type.item()
            next_mol = apply_action(current_mol, action_type_idx, target_node_idx, new_atom_type_idx, state_data.original_indices)

            # Determine reward and done state
            intermediate_reward = 0.0 # Use final reward only
            current_step_in_traj += 1

            if action_type_idx == ACTION_TYPES.index("STOP") or next_mol is None or current_step_in_traj >= num_refinement_steps:
                traj_done = True
                # Calculate reward based on the state *before* STOP/failure
                final_smiles = Chem.MolToSmiles(current_mol)
                if final_smiles:
                    final_reward, _ = calculate_reward(final_smiles, property_predictors)
                    iteration_final_smiles.append(final_smiles)
                else:
                    final_reward = -1.0 # Penalize if final mol invalid
                iteration_final_rewards.append(final_reward)
                current_mol = None # Reset for next rollout start
            else:
                traj_done = False
                final_reward = 0.0 # No final reward yet
                current_mol = next_mol # Continue trajectory

            buffer_rewards.append(final_reward if traj_done else intermediate_reward)
            buffer_dones.append(traj_done)

            collected_steps += 1
            global_step += 1


        # --- 4b. Calculate Advantages and Returns ---
        with torch.no_grad():
            if current_mol: # Trajectory didn't finish, need bootstrap value
                last_state_data = mol_to_pyg_data(current_mol, device=device)
                if last_state_data:
                    last_state_batch = Batch.from_data_list([last_state_data]).to(device)
                    last_value = critic(last_state_batch).cpu()
                else: last_value = torch.tensor([[0.0]]) # Failed conversion
            else: # Trajectory finished naturally
                last_value = torch.tensor([[0.0]]) # Terminal state value is 0

        buffer_values_for_gae = buffer_values + [last_value]
        # Ensure dones align with rewards/values (length should be len(rewards))
        # Need to handle case where buffer ended mid-traj but last state was invalid
        dones_for_gae = buffer_dones
        if len(dones_for_gae) < len(buffer_rewards):
             dones_for_gae.append(True) # Assume done if we added dummy reward/state

        # Ensure lengths match before GAE
        if len(buffer_rewards) == len(buffer_values_for_gae) - 1 and len(buffer_rewards) == len(dones_for_gae):
            advantages = calculate_gae(buffer_rewards, buffer_values_for_gae, dones_for_gae, gamma, gae_lambda, device='cpu')
            returns = advantages + torch.cat(buffer_values).squeeze().cpu() # Calculate returns on CPU
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages
        else:
             logger.warning(f"Length mismatch before GAE: R={len(buffer_rewards)}, V={len(buffer_values_for_gae)}, D={len(dones_for_gae)}. Skipping GAE calculation.")
             # Fallback: use simple rewards as advantages? Risky.
             advantages = torch.tensor(buffer_rewards, dtype=torch.float32).cpu()
             returns = advantages # Very basic fallback
             if len(advantages) != len(buffer_values):
                  logger.error("Catastrophic length mismatch. Aborting PPO update.")
                  continue # Skip PPO update for this iteration


        # --- 4c. PPO Update Phase ---
        # ... (PPO update logic remains largely the same) ...
        actor.train()
        critic.train()

        num_samples = len(buffer_states)
        if num_samples == 0: continue # No valid steps collected
        indices = np.arange(num_samples)

        approx_kl_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        update_count = 0

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                if len(mb_indices) == 0: continue

                # Slice buffer lists carefully
                mb_states_list = [buffer_states[i] for i in mb_indices if buffer_states[i] is not None]
                if not mb_states_list: continue # Skip if all states in minibatch were None
                mb_batch = Batch.from_data_list(mb_states_list).to(device)

                # Ensure actions/logprobs/etc correspond to the valid states
                valid_mb_indices = [i for i in mb_indices if buffer_states[i] is not None]
                if not valid_mb_indices: continue

                # Adjust slicing based on actual valid indices
                mb_actions_type = torch.stack([buffer_actions_type[i] for i in valid_mb_indices]).to(device)
                mb_actions_target = torch.stack([buffer_actions_target[i] for i in valid_mb_indices]).to(device)
                mb_actions_new_atom = torch.stack([buffer_actions_new_atom[i] for i in valid_mb_indices]).to(device)
                mb_old_log_probs_type = torch.stack([buffer_log_probs_type[i] for i in valid_mb_indices]).to(device)
                mb_old_log_probs_target = torch.stack([buffer_log_probs_target[i] for i in valid_mb_indices]).to(device)
                mb_old_log_probs_new_atom = torch.stack([buffer_log_probs_new_atom[i] for i in valid_mb_indices]).to(device)

                # Ensure advantages and returns are indexed correctly
                # Note: advantages/returns were calculated based on *all* buffer steps
                # We need to map valid_mb_indices back to the advantage/return tensors
                mb_advantages = advantages[valid_mb_indices].to(device)
                mb_returns = returns[valid_mb_indices].to(device)


                # Evaluate current policy on minibatch states
                new_action_type_dist, new_target_node_dist_full, new_atom_type_dist, _ = actor(mb_batch)
                new_values = critic(mb_batch).squeeze() # Ensure shape matches returns

                # Calculate log probs for the actions taken
                new_log_probs_type = new_action_type_dist.log_prob(mb_actions_type)

                # --- Target Node Log Prob Calculation ---
                new_log_probs_target_list = []
                node_counts = torch.bincount(mb_batch.batch)
                current_node_idx = 0
                action_idx_in_mb = 0 # Track which action corresponds to which graph
                for i in range(mb_batch.num_graphs):
                     num_nodes_in_graph = node_counts[i].item()
                     end_node_idx = current_node_idx + num_nodes_in_graph
                     graph_node_logits = new_target_node_dist_full.logits[current_node_idx:end_node_idx]

                     if graph_node_logits.nelement() > 0:
                         graph_target_node_dist = Categorical(logits=graph_node_logits)
                         # Get the action for this specific graph in the minibatch
                         graph_action_target = mb_actions_target[action_idx_in_mb]
                         new_log_probs_target_list.append(graph_target_node_dist.log_prob(graph_action_target))
                     else:
                         new_log_probs_target_list.append(torch.tensor(0.0, device=device))

                     current_node_idx = end_node_idx
                     action_idx_in_mb += 1 # Move to the next action in the minibatch sequence
                new_log_probs_target = torch.stack(new_log_probs_target_list)
                # --- End Target Node Log Prob ---

                new_log_probs_new_atom = new_atom_type_dist.log_prob(mb_actions_new_atom)

                # Combine log probs
                mb_old_log_probs = mb_old_log_probs_type + mb_old_log_probs_target + mb_old_log_probs_new_atom
                new_log_probs = new_log_probs_type + new_log_probs_target + new_log_probs_new_atom

                # Calculate policy ratio and PPO loss
                logratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(logratio)
                clip_adv = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()

                # Value loss
                value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_type = new_action_type_dist.entropy().mean()
                entropy_target = new_target_node_dist_full.entropy().mean() # Approx average
                entropy_new_atom = new_atom_type_dist.entropy().mean()
                entropy_bonus = (entropy_type + entropy_target + entropy_new_atom) / 3.0

                # Total loss
                loss = policy_loss - entropy_coef * entropy_bonus + value_loss_coef * value_loss

                # Optimization step
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                optimizer_actor.step()
                optimizer_critic.step()

                # Accumulate stats for logging/early stopping
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                entropy_accum += entropy_bonus.item()
                approx_kl_accum += (mb_old_log_probs - new_log_probs).mean().item() # Accumulate KL estimate
                update_count += 1


            # Check KL divergence early stopping after each epoch
            avg_kl_epoch = approx_kl_accum / update_count if update_count > 0 else 0
            if target_kl is not None and avg_kl_epoch > target_kl:
                logger.info(f"  Epoch {epoch+1}: KL divergence ({avg_kl_epoch:.4f}) exceeded target ({target_kl:.4f}). Stopping PPO updates.")
                break

        # --- Clear Buffer after PPO updates ---
        buffer_states.clear()
        buffer_actions_type.clear()
        buffer_actions_target.clear()
        buffer_actions_new_atom.clear()
        buffer_log_probs_type.clear()
        buffer_log_probs_target.clear()
        buffer_log_probs_new_atom.clear()
        buffer_rewards.clear()
        buffer_values.clear()
        buffer_dones.clear()
        buffer_original_indices.clear()

        # --- Logging ---
        avg_final_reward = np.mean(iteration_final_rewards) if iteration_final_rewards else -1.0
        avg_policy_loss = policy_loss_accum / update_count if update_count > 0 else 0
        avg_value_loss = value_loss_accum / update_count if update_count > 0 else 0
        avg_entropy = entropy_accum / update_count if update_count > 0 else 0
        logger.info(f"Iter [{iteration+1}/{num_iterations}] Steps: {global_step} | Avg Final Reward: {avg_final_reward:.4f} | Avg PLoss: {avg_policy_loss:.4f} | Avg VLoss: {avg_value_loss:.4f} | Avg Entropy: {avg_entropy:.4f}")
        # Log some final SMILES from the iteration
        if iteration_final_smiles:
             logger.info(f"  Sample Final SMILES: {iteration_final_smiles[:min(5, len(iteration_final_smiles))]}")

    # --- 5. Save Trained Models ---
    # ... (Saving logic remains the same) ...
    logger.info("GCPN training finished.")
    os.makedirs(os.path.dirname(actor_save_path), exist_ok=True)
    torch.save(actor.state_dict(), actor_save_path)
    os.makedirs(os.path.dirname(critic_save_path), exist_ok=True)
    torch.save(critic.state_dict(), critic_save_path)
    logger.info(f"Actor saved to: {actor_save_path}")
    logger.info(f"Critic saved to: {critic_save_path}")


# --- Placeholder for Property Predictor Loading ---
# You need to replace this with your actual loading logic
def load_property_predictors(predictor_paths: Dict[str, str], device: torch.device) -> Dict[str, Any]:
    """ Placeholder function to load property prediction models. """
    predictors = {}
    logger.info("Attempting to load property predictors (using placeholders)...")
    for prop_name, path in predictor_paths.items():
        try:
            # Example: Load a dummy predictor or a real model if path exists
            # if os.path.exists(path):
            #     model = torch.load(path, map_location=device)
            #     model.eval()
            #     predictors[prop_name] = model
            #     logger.info(f"  Loaded predictor for '{prop_name}' from {path}")
            # else:
            #     logger.warning(f"  Predictor path not found for '{prop_name}': {path}. Using placeholder.")
            #     predictors[prop_name] = None # Indicate placeholder (reward fn will use RDKit defaults)
            logger.warning(f"  Predictor loading not implemented for '{prop_name}'. Using placeholder (None).")
            predictors[prop_name] = None # Reward function will use RDKit defaults if predictor is None
        except Exception as e:
            logger.error(f"  Failed to load predictor for '{prop_name}' from {path}: {e}")
            predictors[prop_name] = None # Fallback
    return predictors

# Add this helper function to create sample molecules when VAE fails
def create_sample_molecule() -> Optional[Chem.Mol]:
    """Create a simple valid molecule when VAE sampling fails"""
    try:
        # Create a few simple molecules
        samples = [
            'C', 'CC', 'CCC', 'CCCC', 'c1ccccc1', 'CCO', 'c1ccccc1O',
            'CC(=O)O', 'CCN', 'CC=CC', 'CN', 'CO'
        ]
        smiles = np.random.choice(samples)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logger.error(f"Error creating sample molecule: {e}")
        return None


if __name__ == '__main__':
    if not PYG_AVAILABLE or not RDKIT_AVAILABLE:
        logger.error("Missing PyTorch Geometric or RDKit. Cannot run GCPN example.")
    else:
        # --- Example Usage ---
        # ** CRITICAL: Set correct VAE path **
        # ** CRITICAL: Verify all VAE arguments in optimize_gcpn match your trained model **
        # ** CRITICAL: Verify ATOM_MAP/BOND_MAP match VAE preprocessing **
        # ** CRITICAL: Provide paths to real property predictors or update reward function **
        
        # Updated paths to be more flexible
        import os.path
        
        # Try different possible locations for the VAE model
        possible_vae_paths = [
            'models/vae_checkpoints/final_vae_model.pt',   # Actual path from training output
            'experiments/graph_vae_model.pth',             # Original path
            '../experiments/graph_vae_model.pth',          # One directory up
            'models/graph_vae_model.pth',                  # Models directory
            '../models/graph_vae_model.pth',               # Up and models
            'src/models/graph_vae_model.pth',              # In src/models
            'checkpoints/graph_vae_model.pth',             # Checkpoints directory
            'saved_models/graph_vae_model.pth',            # Saved models directory
            './graph_vae_model.pth'                        # Current directory
        ]
        
        # Find the first path that exists
        VAE_MODEL_FILE = None
        for path in possible_vae_paths:
            if os.path.exists(path):
                VAE_MODEL_FILE = path
                logger.info(f"Found VAE model at: {path}")
                break
                
        if VAE_MODEL_FILE is None:
            logger.error("Could not find VAE model file in any of the expected locations.")
            logger.error("Please manually specify the correct path to your trained VAE model.")
            logger.error("Expected paths checked: " + ", ".join(possible_vae_paths))
            logger.error("You can place the model in one of these locations or update the path in this script.")
            exit(1)
            
        # Create output directories if they don't exist
        os.makedirs('experiments', exist_ok=True)
        ACTOR_SAVE_FILE = 'experiments/gcpn_actor_final.pth'
        CRITIC_SAVE_FILE = 'experiments/gcpn_critic_final.pth'
        # Add paths to your actual property predictor models here
        PROPERTY_PREDICTORS_PATHS = {
            # 'qed': 'path/to/your/qed_predictor.pth',
            # 'logp': 'path/to/your/logp_predictor.pth',
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run the optimization
        optimize_gcpn(
            vae_model_path=VAE_MODEL_FILE,
            actor_save_path=ACTOR_SAVE_FILE,
            critic_save_path=CRITIC_SAVE_FILE,
            property_predictor_paths=PROPERTY_PREDICTORS_PATHS,
            # VAE args are defined inside the function now, verify them there
            # --- GCPN/RL Params (tune these) ---
            gcpn_hidden_dim=256,          # UPDATED: Changed from 128 to 256 to match VAE
            gcpn_num_layers=3,
            num_refinement_steps=5,
            num_iterations=60,            # REDUCED: From 500 to 60 for faster completion
            steps_per_iteration=256,      # REDUCED: From 1024 to 256 for faster iterations
            batch_size=32,                # REDUCED: From 64 to 32 for faster updates
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,    # Often same as actor LR
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            ppo_epochs=5,                 # REDUCED: From 10 to 5 for faster updates
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.015,
            device=device
        )

