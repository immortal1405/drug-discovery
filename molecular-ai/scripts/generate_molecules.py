# molecular-ai/scripts/generate_molecules.py

import torch
import json
import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
import logging

# Suppress RDKit warnings/errors during molecule construction
RDLogger.DisableLog('rdApp.*')

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.graph_vae import GraphVAE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Decoding Helpers (based on preprocess_data.py) ---
# These indices need to EXACTLY match the feature creation in preprocess_data.py
ATOM_TYPE_SLICE = slice(0, 100)
DEGREE_SLICE = slice(100, 106)
FORMAL_CHARGE_SLICE = slice(106, 111) # Indices map to charges -2, -1, 0, 1, 2
HYBRIDIZATION_SLICE = slice(111, 116) # sp, sp2, sp3, sp3d, sp3d2
ADDITIONAL_SLICE = slice(116, 120) # IsAromatic, NumRadicalElectrons, TotalNumHs, IsInRing

HYBRIDIZATION_MAP = {
    0: Chem.HybridizationType.SP,
    1: Chem.HybridizationType.SP2,
    2: Chem.HybridizationType.SP3,
    3: Chem.HybridizationType.SP3D,
    4: Chem.HybridizationType.SP3D2
}

FORMAL_CHARGE_MAP = {
    0: -2,
    1: -1,
    2: 0,
    3: 1,
    4: 2
}

def matrices_to_mol(node_features_pred: np.ndarray, adj_logits: np.ndarray, bond_type_logits: np.ndarray) -> Chem.Mol | None:
    """
    Converts predicted node features, adjacency logits, and bond type logits into an RDKit Mol object.

    Args:
        node_features_pred: Predicted node features [max_nodes, num_features]
        adj_logits: Predicted adjacency logits [max_nodes, max_nodes]
        bond_type_logits: Predicted bond type logits [max_nodes, max_nodes, num_bond_types]

    Returns:
        An RDKit Mol object or None.
    """
    adj_threshold = 0.5 # Use a standard threshold now
    logger.info(f"Using adjacency threshold: {adj_threshold}")
    num_bond_types = 4 # SINGLE, DOUBLE, TRIPLE, AROMATIC - Ensure matches decoder/loss
    rdkit_bond_types = [
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
    ]
    try:
        max_nodes, num_features = node_features_pred.shape
        
        # Slices for feature decoding (ensure these are correct for 120 features)
        ATOM_TYPE_SLICE = slice(0, 100) 
        DEGREE_SLICE = slice(100, 106)
        FORMAL_CHARGE_SLICE = slice(106, 111)
        HYBRIDIZATION_SLICE = slice(111, 116)
        ADDITIONAL_SLICE = slice(116, 120)
        # Map for formal charge
        FORMAL_CHARGE_MAP = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4} # Value: Index
        FORMAL_CHARGE_MAP = {v: k for k, v in FORMAL_CHARGE_MAP.items()} # Invert: Index: Value

        # --- Adjacency Probabilities ---
        adj_prob = torch.sigmoid(torch.from_numpy(adj_logits)).numpy()
        adj_binary = (adj_prob > adj_threshold).astype(int)
        adj_binary = np.triu(adj_binary, k=1) # Take upper triangle, excluding diagonal
        adj_binary = adj_binary + adj_binary.T # Make symmetric
        
        # --- Bond Type Probabilities/Predictions ---
        bond_type_probs = torch.softmax(torch.from_numpy(bond_type_logits), dim=-1).numpy() # Shape [N, N, C]
        predicted_bond_indices = np.argmax(bond_type_probs, axis=-1) # Shape [N, N]

        # Determine actual number of nodes (simple heuristic: assume all)
        num_effective_nodes = max_nodes

        # Create editable molecule
        mol = Chem.RWMol()

        # 2. Add Atoms
        added_atom_indices = {} # Map matrix index to RDKit index
        rdkit_idx_counter = 0
        for i in range(num_effective_nodes):
            features = node_features_pred[i]
            
            # Apply Softmax before argmax for categorical features
            atom_type_probs = torch.softmax(torch.from_numpy(features[ATOM_TYPE_SLICE]), dim=0).numpy()
            charge_probs = torch.softmax(torch.from_numpy(features[FORMAL_CHARGE_SLICE]), dim=0).numpy()
            # hybridization_probs = torch.softmax(torch.from_numpy(features[HYBRIDIZATION_SLICE]), dim=0).numpy()

            # Decode Atom Type 
            atom_type_idx = np.argmax(atom_type_probs)
            atomic_num = atom_type_idx + 1
            if not (1 <= atomic_num <= 100): continue 

            atom = Chem.Atom(atomic_num)

            # Decode Formal Charge
            charge_idx = np.argmax(charge_probs)
            formal_charge = FORMAL_CHARGE_MAP.get(charge_idx, 0) 
            atom.SetFormalCharge(formal_charge)
            
            # Add atom to molecule
            rdkit_idx = mol.AddAtom(atom)
            added_atom_indices[i] = rdkit_idx 
            rdkit_idx_counter += 1

        if rdkit_idx_counter == 0: return None # No valid atoms added

        # 3. Add Bonds using predicted types
        for i in range(num_effective_nodes):
            for j in range(i + 1, num_effective_nodes):
                if adj_binary[i, j] == 1: # If edge exists based on adjacency prediction
                    if i in added_atom_indices and j in added_atom_indices:
                        idx1 = added_atom_indices[i]
                        idx2 = added_atom_indices[j]

                        # Use predicted bond type
                        bond_type_idx = predicted_bond_indices[i, j]
                        if 0 <= bond_type_idx < len(rdkit_bond_types):
                            predicted_type = rdkit_bond_types[bond_type_idx]
                        else:
                            predicted_type = Chem.BondType.SINGLE # Fallback

                        # Avoid adding duplicate bonds
                        if mol.GetBondBetweenAtoms(idx1, idx2) is None:
                            mol.AddBond(idx1, idx2, predicted_type)

        # 4. Sanitize and Validate
        Chem.SanitizeMol(mol) # Attempt to fix valencies, aromaticity etc.
        
        # Check if molecule is valid after sanitization
        smiles = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(smiles) is None: # Final check
             # logger.debug(f"Post-sanitize check failed for SMILES derived from generated mol.")
             return None
             
        return mol

    except Exception as e:
        # logger.debug(f"Error converting matrices to Mol: {e}")
        return None

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Generate Molecules from Trained Graph VAE')
    parser.add_argument('--config_path', type=str, default='config/training_config.json', help='Path to the training configuration file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the VAE model checkpoint (.pt file).')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of molecules to generate.')
    parser.add_argument('--output_file', type=str, default='generated_smiles.txt', help='File to save generated valid SMILES strings.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generation.')
    # Add threshold for adjacency matrix if needed
    # parser.add_argument('--adj_threshold', type=float, default=0.5, help='Threshold for converting adjacency probabilities to bonds.')
    return parser

def load_model_and_config(args):
    # Load configuration
    config_path = os.path.join(project_root, args.config_path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        sys.exit(1)
        
    vae_model_cfg = config.get('vae_model_config')
    if not vae_model_cfg:
        logger.error("'vae_model_config' not found in configuration file.")
        sys.exit(1)
        
    # **Important**: Determine node_features_dim. 
    # Use the value from the config file, which should match training.
    node_features_dim = vae_model_cfg.get('node_features', None) # Try vae_model_config first
    if node_features_dim is None:
        # Attempt to get from general model config as secondary fallback
        logger.warning("'node_features' not found in 'vae_model_config', checking 'model_config'.")
        node_features_dim = config.get('model_config', {}).get('node_features') 
    
    if node_features_dim is None:
        logger.error("Could not determine 'node_features' dimension from config.")
        sys.exit(1)
    logger.info(f"Using node_features dimension from config: {node_features_dim}")

    # Initialize model
    model = GraphVAE(
        node_features=node_features_dim, # Use dimension determined from config
        hidden_dim=vae_model_cfg['hidden_dim'],
        latent_dim=vae_model_cfg['latent_dim'],
        max_nodes=vae_model_cfg['max_nodes'],
        num_enc_layers=vae_model_cfg['num_enc_layers'],
        heads_enc=vae_model_cfg['heads_enc'],
        dropout_enc=vae_model_cfg['dropout_enc'],
        num_dec_layers=vae_model_cfg['num_dec_layers'],
        heads_dec=vae_model_cfg['heads_dec'],
        dropout_dec=vae_model_cfg['dropout_dec'],
        lora_r=vae_model_cfg.get('lora_r', 0), # Use get for potential absence in older configs
        lora_alpha=vae_model_cfg.get('lora_alpha', 1),
        lora_dropout=vae_model_cfg.get('lora_dropout', 0)
    ).to(args.device)

    # Load checkpoint
    checkpoint_full_path = os.path.join(project_root, args.checkpoint_path)
    try:
        model.load_state_dict(torch.load(checkpoint_full_path, map_location=args.device))
        logger.info(f"Loaded model state dict from {checkpoint_full_path}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at {checkpoint_full_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        sys.exit(1)
        
    model.eval() # Set model to evaluation mode
    return model, vae_model_cfg

def generate_and_evaluate(model, cfg, args):
    logger.info(f"Starting generation of {args.num_samples} samples...")
    all_smiles = []
    valid_smiles = []
    generated_count = 0

    with torch.no_grad():
        while generated_count < args.num_samples:
            batch_to_generate = min(args.batch_size, args.num_samples - generated_count)
            if batch_to_generate <= 0:
                break
            
            # Sample z from prior
            z = torch.randn(batch_to_generate, cfg['latent_dim']).to(args.device)
            
            # Decode - model now returns bond types too
            node_features_pred, adj_logits_pred, bond_type_logits_pred = model.decoder(z)
            
            # --- Debugging: Print raw output for the first sample and exit ---
            if generated_count == 0:
                logger.info("--- Raw Decoder Output (Sample 0) ---")
                logger.info(f"Node Features Shape: {node_features_pred[0].shape}")
                logger.info(f"Node Features (Sample 0, Node 0):\n{node_features_pred[0, 0, :].cpu().numpy()}") # Print features of first node
                logger.info(f"Adjacency Logits Shape: {adj_logits_pred[0].shape}")
                logger.info(f"Adjacency Logits (Sample 0, Partial):\n{adj_logits_pred[0, :5, :5].cpu().numpy()}") # Print top-left 5x5 corner
                # Optional: Add more detailed printing here if needed
                # logger.info("Exiting after printing debug info.")
                # sys.exit(0) # Exit after first batch if uncommented
            # --- End Debugging ---

            # Process batch
            for i in range(batch_to_generate):
                # Pass bond type logits to conversion function
                mol = matrices_to_mol(node_features_pred[i].cpu().numpy(), 
                                      adj_logits_pred[i].cpu().numpy(), 
                                      bond_type_logits_pred[i].cpu().numpy()) 
                if mol:
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        all_smiles.append(smiles)
                        # Extra check for validity via re-parsing
                        if Chem.MolFromSmiles(smiles): 
                            valid_smiles.append(smiles)
                    except Exception:
                        pass # Ignore SMILES conversion errors

            generated_count += batch_to_generate
            logger.info(f"Generated {generated_count}/{args.num_samples}...")

    logger.info(f"Finished generation.")
    
    # --- Evaluation Metrics ---
    num_attempted = args.num_samples # Could also use len(all_smiles) if conversion fails often
    num_valid = len(valid_smiles)
    validity = (num_valid / num_attempted) * 100 if num_attempted > 0 else 0
    
    unique_valid_smiles = set(valid_smiles)
    num_unique = len(unique_valid_smiles)
    uniqueness = (num_unique / num_valid) * 100 if num_valid > 0 else 0
    
    logger.info(f"--- Evaluation Results ---")
    logger.info(f"Attempted Generations: {num_attempted}")
    logger.info(f"Valid Molecules: {num_valid}")
    logger.info(f"Unique Valid Molecules: {num_unique}")
    logger.info(f"Validity: {validity:.2f}%")
    logger.info(f"Uniqueness (among valid): {uniqueness:.2f}%")
    
    # Save valid SMILES
    output_full_path = os.path.join(project_root, args.output_file)
    with open(output_full_path, 'w') as f:
        for smiles in unique_valid_smiles:
            f.write(smiles + '\n')
    logger.info(f"Saved {num_unique} unique valid SMILES to {output_full_path}")

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Ensure the checkpoint path is interpreted relative to project root if needed
    # Note: Checkpoint paths from training might already be relative like 'models/vae_checkpoints/...'
    # No change needed if args.checkpoint_path is like 'models/vae_checkpoints/best_vae_model_epoch_86.pt'
    
    model, config = load_model_and_config(args)
    generate_and_evaluate(model, config, args)
    logger.info("Generation script finished.") 