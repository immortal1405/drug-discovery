import torch
import json
import os
import sys
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader # To handle potential batching if needed

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.mpnn import MolecularMPNN
from preprocess_data import MolecularDataset 

def visualize_attention(model: MolecularMPNN, data: Data, layer_idx: int = -1, head_idx: int = 0, filename: str = "attention_visualization.png"):
    """
    Visualizes attention weights for a single molecule using RDKit.

    Args:
        model: The trained MolecularMPNN model.
        data: The PyTorch Geometric Data object for the molecule.
        layer_idx: Index of the GAT layer whose attention to visualize (-1 for last layer).
        head_idx: Index of the attention head to visualize.
        filename: Path to save the visualization image.
    """
    model.eval()
    with torch.no_grad():
        # The model now returns predictions and a list of attention tuples
        predictions, attention_list = model(data)

    if not attention_list or layer_idx >= len(attention_list):
        print(f"Error: Attention weights not available or invalid layer index {layer_idx}.")
        return

    # Get attention weights for the specified layer
    edge_index, attention_weights = attention_list[layer_idx] # attention_weights shape: [num_edges, num_heads]

    if head_idx >= attention_weights.shape[1]:
        print(f"Error: Invalid head index {head_idx}. Max index: {attention_weights.shape[1]-1}")
        return

    # Select weights for the specified head
    attn_weights_head = attention_weights[:, head_idx].cpu().numpy() # Shape: [num_edges]

    # Ensure edge_index is on CPU and Long type for indexing
    edge_index = edge_index.cpu()

    # --- RDKit Visualization ---
    smiles = data.smiles # Assuming smiles is stored in the data object
    if not smiles:
        print("Error: SMILES string not found in data object.")
        return
        
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Error: Could not create RDKit molecule from SMILES: {smiles}")
        return

    num_atoms = mol.GetNumAtoms()
    num_edges = edge_index.shape[1]

    if num_edges != len(attn_weights_head):
         print(f"Warning: Number of edges in graph ({num_edges}) doesn't match attention weights ({len(attn_weights_head)}). Visualization might be incorrect.")
         # Attempt to proceed, but might fail or be misleading

    # Map edge attention weights to atoms (e.g., sum attention for edges connected to an atom)
    atom_attention = [0.0] * num_atoms
    for i in range(num_edges):
        src_node = edge_index[0, i].item()
        tgt_node = edge_index[1, i].item()
        weight = attn_weights_head[i]
        
        # Add attention to both source and target nodes involved in the edge
        if src_node < num_atoms:
             atom_attention[src_node] += weight
        if tgt_node < num_atoms:
             atom_attention[tgt_node] += weight
             
    # Normalize attention weights for visualization (optional, helps consistency)
    max_attn = max(atom_attention) if atom_attention else 1.0
    norm_atom_attention = [attn / max_attn for attn in atom_attention] if max_attn > 0 else atom_attention


    # Create image with attention highlights
    img = Draw.MolToImage(mol, size=(600, 600), highlightAtoms=list(range(num_atoms)),
                          highlightAtomColors={i: plt.cm.viridis(norm_atom_attention[i]) for i in range(num_atoms)})

    img.save(filename)
    print(f"Attention visualization saved to {filename}")


if __name__ == "__main__":
    # --- Configuration ---
    config_path = os.path.join(project_root, 'config', 'training_config.json')
    model_path = os.path.join(project_root, 'models', 'best_model.pt')
    test_data_path = os.path.join(project_root, 'data', 'processed', 'test.pt')
    output_filename = "attention_visualization.png"
    molecule_index = 0 # Index of the molecule in the test set to visualize
    layer_to_visualize = -1 # Last GAT layer
    head_to_visualize = 0 # First attention head

    # --- Load Config ---
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    # Use dimensions detected during training that created the saved model
    detected_node_features = 120
    detected_edge_features = 6
    print(f"Initializing model with detected features: Node={detected_node_features}, Edge={detected_edge_features}")
    
    model = MolecularMPNN(
        node_features=detected_node_features, # Use detected features
        edge_features=detected_edge_features, # Use detected features
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_tasks=model_config['num_tasks'],
        heads=model_config.get('heads', 4) # Use configured heads, default 4
    )
    
    try:
        # Adjust loading based on how the model was saved (state_dict vs entire model)
        model_state = torch.load(model_path, map_location=device)
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
             model.load_state_dict(model_state['model_state_dict'])
        else:
             # Assuming the entire model object was saved
             model.load_state_dict(model_state) 
        model.to(device)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

    # --- Load Data ---
    try:
        # Load the processed data using the correct dataset class
        # Note: The MolecularDataset in preprocess_data expects a pre-processed .pt file,
        # so we load the file directly instead of initializing the dataset class to process raw data.
        # Set weights_only=False to load pickled Data objects
        test_data_list = torch.load(test_data_path, map_location=device, weights_only=False)
        
        if not isinstance(test_data_list, list):
             print(f"Error: Expected loaded data to be a list, but got {type(test_data_list)}.")
             sys.exit(1)

        if molecule_index >= len(test_data_list):
            print(f"Error: molecule_index {molecule_index} is out of bounds for test dataset size {len(test_data_list)}.")
            sys.exit(1)
            
        # Get the specific molecule data object
        molecule_data = test_data_list[molecule_index]
        
        # Ensure data is on the correct device
        molecule_data = molecule_data.to(device) 

        # Add batch dimension for model input
        # Create a simple Data object if needed or use DataLoader with batch_size=1
        # For simplicity, let's assume the model can handle a single Data object if it's structured correctly
        # Often requires adding a 'batch' attribute:
        molecule_data.batch = torch.zeros(molecule_data.num_nodes, dtype=torch.long, device=device)
        
        # Fetch actual node/edge features dimensions from loaded data
        actual_node_features = molecule_data.num_node_features
        actual_edge_features = molecule_data.num_edge_features
        
        # Validate model config against data dimensions
        if model_config['node_features'] != actual_node_features:
             print(f"Warning: Config node_features ({model_config['node_features']}) mismatch data ({actual_node_features}). Model may fail.")
             # Optionally update model_config if you want to load based on data
             # model_config['node_features'] = actual_node_features 
        if model_config['edge_features'] != actual_edge_features:
             print(f"Warning: Config edge_features ({model_config['edge_features']}) mismatch data ({actual_edge_features}). Model may fail.")
             # model_config['edge_features'] = actual_edge_features

    except FileNotFoundError:
        print(f"Error: Processed test dataset not found at {test_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        sys.exit(1)


    # --- Visualize ---
    if hasattr(molecule_data, 'smiles'):
         visualize_attention(model, molecule_data, layer_idx=layer_to_visualize, head_idx=head_to_visualize, filename=output_filename)
    else:
         print("Error: Cannot visualize. SMILES string missing from the loaded data object.") 