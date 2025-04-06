import torch
import json
import os
import sys
import numpy as np
# Import DeepLift -> Remove this
# from captum.attr import DeepLift
import shap # Import shap library
import matplotlib.pyplot as plt # For coloring
import matplotlib.colors as mcolors # For normalization
from rdkit import Chem
from rdkit.Chem import Draw

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.mpnn import MolecularMPNN
# Import Data for constructing object inside predict_fn
from torch_geometric.data import Data

# Define feature names based on preprocess_data.py (manual calculation part)
ATOM_TYPE_FEATURES = [f"AtomType_{i+1}" for i in range(100)]
DEGREE_FEATURES = [f"Degree_{i}" for i in range(6)]
FORMAL_CHARGE_FEATURES = [f"FormalCharge_{c}" for c in range(-2, 3)] # Indices 0 to 4 map to -2 to +2
HYBRIDIZATION_FEATURES = ['sp', 'sp2', 'sp3', 'sp3d', 'sp3d2']
ADDITIONAL_ATOM_FEATURES = ['IsAromatic', 'NumRadicalElectrons', 'TotalNumHs', 'IsInRing']

NODE_FEATURE_NAMES = (
    ATOM_TYPE_FEATURES +
    DEGREE_FEATURES +
    FORMAL_CHARGE_FEATURES +
    HYBRIDIZATION_FEATURES +
    ADDITIONAL_ATOM_FEATURES
)

if len(NODE_FEATURE_NAMES) != 120:
     print(f"Warning: Number of feature names ({len(NODE_FEATURE_NAMES)}) does not match expected 120 features.")
     # Pad or truncate if necessary, or fix the names list
     NODE_FEATURE_NAMES = NODE_FEATURE_NAMES[:120] + [f"Unknown_{i}" for i in range(120 - len(NODE_FEATURE_NAMES))]


# --- Remove ModelWrapper --- 

# Prediction function for shap.KernelExplainer
# Declare these variables globally so predict_fn can access them
molecule_data = None
model_orig = None
device = None
TARGET_TASK_INDEX = -1 # Will be set in main
detected_node_features = -1 # Will be set in main

def predict_fn(x_perturbed_np):
    """
    Takes a batch of perturbed node feature matrices (numpy) and returns model predictions.
    x_perturbed_np shape: [num_samples, num_nodes * num_features] or [num_samples, num_nodes, num_features]
    """
    if molecule_data is None or model_orig is None or device is None or TARGET_TASK_INDEX == -1 or detected_node_features == -1:
        raise ValueError("Global variables for predict_fn not set properly in main block.")

    num_samples = x_perturbed_np.shape[0]
    predictions_list = []

    # Reshape if SHAP provides flattened features per node
    if x_perturbed_np.ndim == 2 and x_perturbed_np.shape[1] == (molecule_data.num_nodes * detected_node_features):
         x_perturbed_np = x_perturbed_np.reshape(num_samples, molecule_data.num_nodes, detected_node_features)
    elif x_perturbed_np.ndim != 3 or x_perturbed_np.shape[1:] != (molecule_data.num_nodes, detected_node_features):
         raise ValueError(f"Unexpected input shape to predict_fn: {x_perturbed_np.shape}")

    # Convert numpy array to torch tensor
    x_perturbed = torch.tensor(x_perturbed_np, dtype=torch.float).to(device)

    with torch.no_grad():
        for i in range(num_samples):
            # Reconstruct Data object for each sample
            temp_data = Data(
                x=x_perturbed[i], # Use perturbed features for this sample
                edge_index=molecule_data.edge_index, # Original connectivity
                batch=molecule_data.batch # Original batch info (for single molecule)
            )
            # Run the original model
            preds, _ = model_orig(temp_data)
            # Get prediction for the target task
            task_pred = preds[:, TARGET_TASK_INDEX].item() # Get single scalar prediction
            predictions_list.append(task_pred)

    return np.array(predictions_list)

def visualize_atom_shap(mol, atom_shap_values, filename="shap_atom_visualization.png"):
    """Visualize per-atom SHAP values on the molecule."""
    num_atoms = mol.GetNumAtoms()
    print(f"[Visualize] RDKit Mol Num Atoms (incl. H): {num_atoms}") # Debug print
    print(f"[Visualize] Length of atom_shap_values: {len(atom_shap_values)}") # Debug print
    
    if len(atom_shap_values) != num_atoms:
        print(f"Error: Number of SHAP values ({len(atom_shap_values)}) doesn't match number of atoms ({num_atoms}). Visualization skipped.")
        return

    # Normalize SHAP values for coloring
    min_val, max_val = np.min(atom_shap_values), np.max(atom_shap_values)
    print(f"Atom SHAP range: [{min_val:.3f}, {max_val:.3f}]") # Keep this print
    
    # Check conditions for TwoSlopeNorm
    if min_val < 0 < max_val:
        # Data spans across zero, use TwoSlopeNorm centered at 0
        norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        print("Using TwoSlopeNorm centered at 0.")
    else:
        # All values are positive or all are negative (or all zero)
        # Use simple Normalize across the actual range
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        print("Using Normalize (all values on one side of 0 or constant).")
        # Add a check for constant value case to avoid vmin=vmax issues with Normalize
        if abs(min_val - max_val) < 1e-9:
             norm = mcolors.Normalize(vmin=min_val - 1e-6, vmax=max_val + 1e-6)
             print(f"Warning: All atom SHAP values are nearly identical ({min_val:.3f}). Adjusting Normalize range.")

    cmap = plt.cm.coolwarm # Red for positive, Blue for negative contribution

    atom_colors = {}
    for i in range(num_atoms):
        atom_colors[i] = cmap(norm(atom_shap_values[i]))

    # Create image with atom highlights based on SHAP values
    # Configure drawing options to add atom indices
    drawOptions = Draw.MolDrawOptions()
    drawOptions.addAtomIndices = True 
    img = Draw.MolToImage(mol, size=(600, 600), highlightAtoms=list(range(num_atoms)),
                          highlightAtomColors=atom_colors, 
                          legend=f'SHAP contributions to LogP\nMin: {min_val:.2f}, Max: {max_val:.2f}',
                          options=drawOptions) # Pass configured options object

    img.save(filename)
    print(f"Per-atom SHAP visualization saved to {filename}")

if __name__ == "__main__":
    # --- Configuration ---
    config_path = os.path.join(project_root, 'config', 'training_config.json')
    model_path = os.path.join(project_root, 'models', 'best_model.pt')
    test_data_path = os.path.join(project_root, 'data', 'processed', 'test.pt')
    molecule_index = 0
    TARGET_TASK_INDEX = 1 # Explain LogP (index 1)
    TOP_N_FEATURES = 15
    KERNEL_SHAP_NSAMPLES = 50 # Can increase for better accuracy, decrease for speed
    output_vis_filename = "shap_logp_visualization.png" # Output filename for visualization

    # --- Load Config ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    detected_node_features = 120 # Assign global
    detected_edge_features = 6   # Based on previous logs
    print(f"Initializing model with detected features: Node={detected_node_features}, Edge={detected_edge_features}")

    model_orig = MolecularMPNN( # Assign global
        node_features=detected_node_features,
        edge_features=detected_edge_features,
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_tasks=model_config['num_tasks'],
        heads=model_config.get('heads', 4)
    )

    try:
        model_state = torch.load(model_path, map_location=device)
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
             model_orig.load_state_dict(model_state['model_state_dict'])
        else:
             model_orig.load_state_dict(model_state)
        model_orig.to(device)
        model_orig.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

    # --- Load Data ---
    try:
        test_data_list = torch.load(test_data_path, map_location=device, weights_only=False)
        if molecule_index >= len(test_data_list):
            print(f"Error: molecule_index {molecule_index} out of bounds.")
            sys.exit(1)
        molecule_data = test_data_list[molecule_index].to(device) # Assign global
        print(f"[Debug] Loaded molecule_data.num_nodes: {molecule_data.num_nodes}") # Debug print
        if not hasattr(molecule_data, 'batch') or molecule_data.batch is None:
             molecule_data.batch = torch.zeros(molecule_data.num_nodes, dtype=torch.long, device=device)
        print(f"Loaded molecule {molecule_index} with SMILES: {molecule_data.smiles}")
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        sys.exit(1)

    # --- Prepare for KernelExplainer ---
    data_to_explain_flat = molecule_data.x.cpu().numpy().reshape(1, -1)
    num_nodes = molecule_data.num_nodes # Assign here for predict_fn scope AND reshaping
    print(f"[Debug] num_nodes used for reshaping/processing: {num_nodes}") # Debug print
    background_data_flat = np.zeros_like(data_to_explain_flat)

    # --- Initialize and Run KernelExplainer ---
    print(f"Initializing KernelExplainer (nsamples={KERNEL_SHAP_NSAMPLES})...")
    # The explainer will call predict_fn with arrays of shape [KERNEL_SHAP_NSAMPLES, num_nodes * num_features]
    explainer = shap.KernelExplainer(predict_fn, background_data_flat)

    # --- Calculate Attributions -> Rename section --- 
    print(f"Calculating SHAP values for Task {TARGET_TASK_INDEX} (LogP)...")
    try:
        shap_values_flat = explainer.shap_values(data_to_explain_flat, nsamples=KERNEL_SHAP_NSAMPLES)
        print(f"[Debug] shap_values_flat.shape: {shap_values_flat.shape}") # Debug print
        
        # Reshape SHAP values back to [num_nodes, num_features]
        # Ensure we use the same num_nodes as determined earlier
        shap_values_per_node = shap_values_flat.reshape(num_nodes, detected_node_features)
        print(f"[Debug] shap_values_per_node.shape: {shap_values_per_node.shape}") # Debug print
        print("SHAP values calculated.")

        # --- Aggregate & Visualize Per-Atom SHAP values ---
        atom_total_shap = shap_values_per_node.sum(axis=1)
        print(f"[Debug] atom_total_shap length: {len(atom_total_shap)}") # Debug print

        # Visualize these values on the molecule structure
        # Generate RDKit molecule WITH hydrogens to match graph nodes
        base_mol = Chem.MolFromSmiles(molecule_data.smiles)
        if base_mol:
            mol = Chem.AddHs(base_mol) # Add hydrogens
            visualize_atom_shap(mol, atom_total_shap, filename=output_vis_filename)
        else:
            print(f"Could not generate RDKit molecule for visualization from SMILES: {molecule_data.smiles}")

        # --- Process Overall Feature Importance (Optional - keep for reference) ---
        feature_importance = np.abs(shap_values_per_node).mean(axis=0)
        if len(feature_importance) != len(NODE_FEATURE_NAMES):
             print(f"Warning: Mismatch between feature importance length ({len(feature_importance)}) and feature names ({len(NODE_FEATURE_NAMES)}).")
        sorted_indices = np.argsort(feature_importance)[::-1]
        print(f"\n--- Top {TOP_N_FEATURES} Node Feature Importances (Mean Abs SHAP) for Task {TARGET_TASK_INDEX} (LogP) ---") # Update task name
        for i in range(min(TOP_N_FEATURES, len(feature_importance))):
            idx = sorted_indices[i]
            feature_name = NODE_FEATURE_NAMES[idx] if idx < len(NODE_FEATURE_NAMES) else f"Feature_{idx}"
            importance_score = feature_importance[idx]
            print(f"{i+1}. {feature_name}: {importance_score:.4f}")

    except Exception as e:
        print(f"Error during SHAP calculation or visualization: {e}") # Adjust print message
        import traceback
        traceback.print_exc() 