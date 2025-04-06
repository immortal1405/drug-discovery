# molecular-ai/scripts/train_vae.py

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime
import time

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset # Keep this for type hint if MolecularDataset is used
from google.cloud import aiplatform

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.graph_vae import GraphVAE
# Assuming MolecularDataset defined in preprocess_data loads the .pt files
from .preprocess_data import MolecularDataset 
from src.utils.data_utils import load_processed_data

# --- Logging Setup ---
log_file = Path("logs") / f"train_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# --- VAE Training Pipeline ---
def train_vae(config: Dict):
    """Trains the Graph VAE model."""
    logger.info("Starting VAE training pipeline...")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Ensure model save directory exists
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    best_model_path = model_dir / "best_vae_model.pt"
    final_model_path = model_dir / "final_vae_model.pt"

    # --- Vertex AI Setup (Optional) ---
    use_vertex_ai = config.get('vertex_ai_config', {}).get('use_vertex_ai', False)
    if use_vertex_ai:
        aiplatform.init(
            project=config['vertex_ai_config']['project_id'],
            location=config['vertex_ai_config']['location']
        )
        experiment_name = config['vertex_ai_config'].get('experiment_name', 'graph-vae-training')
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        aiplatform.start_run(run=run_name)
        aiplatform.log_params(config) # Log hyperparameters
        logger.info(f"Vertex AI experiment {experiment_name} run {run_name} started.")

    # --- Data Loading ---
    try:
        # Adapt based on how MolecularDataset loads preprocessed files
        # Corrected filenames based on ls output
        data_root = os.path.join(project_root, 'data', 'processed')
        # Use the load_processed_data utility function
        train_dataset = load_processed_data(os.path.join(data_root, 'train.pt'))
        val_dataset = load_processed_data(os.path.join(data_root, 'val.pt'))
        
        # Create DataLoader instances
        batch_size = config.get('vae_training_config', {}).get('batch_size', 64) # Use VAE batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        logger.info(f"Loaded {len(train_dataset)} training graphs and {len(val_dataset)} validation graphs.")
        
        # Determine max_nodes from training data if not specified in vae_model_config
        vae_model_cfg = config['vae_model_config'] # Ensure we use the right config section
        if 'max_nodes' not in vae_model_cfg or vae_model_cfg['max_nodes'] <= 0:
            max_nodes_train = max(data.num_nodes for data in train_dataset if hasattr(data, 'num_nodes'))
            max_nodes_val = max(data.num_nodes for data in val_dataset if hasattr(data, 'num_nodes'))
            vae_model_cfg['max_nodes'] = max(max_nodes_train, max_nodes_val)
            logger.info(f"Detected max_nodes: {vae_model_cfg['max_nodes']}")
        # max_nodes = vae_model_cfg['max_nodes'] # Already assigned within the block if needed
            
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data file: {e}. Ensure 'train.pt' and 'val.pt' exist in '{data_root}'. Did you run preprocess_data.py?")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading data: {e}")
        return

    # --- Model Initialization ---
    # Determine node feature dimension from the loaded data
    if train_dataset and len(train_dataset) > 0:
        node_features_dim = train_dataset[0].num_node_features
        logger.info(f"Determined node_features dimension from data: {node_features_dim}")
    else:
        # Fallback or error if data is empty
        logger.error("Training dataset is empty, cannot determine node features dimension.")
        return 
        # node_features_dim = config.get('vae_model_config', {}).get('node_features', 120) # Old fallback

    # node_features_dim = config.get('model_config', {}).get('node_features', 120) # Old incorrect line
    vae_model_cfg = config['vae_model_config']
    logger.info(f"Initializing GraphVAE model with MPNN Encoder + LoRA...")
    logger.info(f"VAE Config: {json.dumps(vae_model_cfg, indent=2)}")

    model = GraphVAE(
        node_features=node_features_dim, # Use dimension determined from data
        hidden_dim=vae_model_cfg['hidden_dim'],
        latent_dim=vae_model_cfg['latent_dim'],
        max_nodes=vae_model_cfg['max_nodes'],
        # Encoder params
        num_enc_layers=vae_model_cfg['num_enc_layers'],
        heads_enc=vae_model_cfg['heads_enc'],       # Renamed arg
        dropout_enc=vae_model_cfg['dropout_enc'],   # Renamed arg
        # Decoder params
        num_dec_layers=vae_model_cfg['num_dec_layers'],
        heads_dec=vae_model_cfg['heads_dec'],
        dropout_dec=vae_model_cfg['dropout_dec'],
        # LoRA params
        lora_r=vae_model_cfg['lora_r'],
        lora_alpha=vae_model_cfg['lora_alpha'],
        lora_dropout=vae_model_cfg['lora_dropout']
    ).to(device)
    
    # --- Optimizer --- 
    # Filter parameters to only train LoRA weights
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]},
    ]
    # Log parameters being optimized
    logger.info("Optimizing the following LoRA parameters:")
    for n, p in model.named_parameters():
         if "lora_" in n and p.requires_grad:
             logger.info(f"  - {n} (shape: {p.shape})")
    # Warn if no LoRA params found
    if not optimizer_grouped_parameters[0]['params']:
         logger.warning("No LoRA parameters found to optimize. Check model structure and LoRA naming.")

    optimizer = optim.Adam(
        optimizer_grouped_parameters, 
        lr=config['vae_training_config'].get('learning_rate', 0.001)
    )
    logger.info(f"Optimizer: Adam, LR: {config['vae_training_config'].get('learning_rate', 0.001)}")
    
    # --- Training Loop --- 
    vae_train_cfg = config['vae_training_config']
    num_epochs = vae_train_cfg.get('num_epochs', 100)
    patience = vae_train_cfg.get('patience', 10)
    # beta = vae_train_cfg.get('beta', 1.0) # Old fixed beta
    checkpoint_dir = vae_train_cfg.get('checkpoint_dir', 'models/vae_checkpoints')
    
    # --- Beta Annealing Setup ---
    anneal_beta = vae_train_cfg.get('anneal_beta', False)
    beta_fixed = vae_train_cfg.get('beta', 1.0) # Fixed beta if annealing is off
    beta_start = vae_train_cfg.get('beta_start', 0.001)
    beta_end = vae_train_cfg.get('beta_end', 1.0)
    beta_anneal_epochs = vae_train_cfg.get('beta_anneal_epochs', 50)
    # --- Free Bits Param ---
    kld_lambda = vae_train_cfg.get('kld_free_bits_lambda', 0.0) 
    # -----------------------

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    logger.info(f"Starting VAE training for {num_epochs} epochs...")
    # logger.info(f"Beta (KLD weight): {beta}") # Log dynamically below
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    if anneal_beta:
        logger.info(f"Beta Annealing: Enabled (Start: {beta_start}, End: {beta_end}, Epochs: {beta_anneal_epochs})")
    else:
        logger.info(f"Beta Annealing: Disabled (Using fixed beta: {beta_fixed})")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        
        # --- Calculate current beta for annealing ---
        if anneal_beta:
            progress = min(1.0, epoch / beta_anneal_epochs)
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_fixed
        # -------------------------------------------
        logger.info(f"Current Beta: {current_beta:.5f}") # Log current beta

        # Pass current_beta and kld_lambda to training and evaluation functions
        train_loss, train_recon, train_kld, train_orig_kld = train_vae_epoch(
            model, train_loader, optimizer, device, current_beta, kld_lambda
        )
        logger.info(f"Train Avg Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KLD (Loss): {train_kld:.4f}, KLD (Orig): {train_orig_kld:.4f}")
        
        val_loss, val_recon, val_kld, val_orig_kld = evaluate_vae(
            model, val_loader, device, current_beta, kld_lambda
        )
        logger.info(f"Val Avg Loss:   {val_loss:.4f}, Recon: {val_recon:.4f}, KLD (Loss): {val_kld:.4f}, KLD (Orig): {val_orig_kld:.4f}")
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch completed in {epoch_duration:.2f} seconds.")

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Clear previous best checkpoints (optional)
            # for f_name in os.listdir(checkpoint_dir):
            #     if f_name.startswith('best_vae_model_'):
            #         os.remove(os.path.join(checkpoint_dir, f_name))
            checkpoint_path = os.path.join(checkpoint_dir, f'best_vae_model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Validation loss improved. Saved model checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # --- Save Final Model --- 
    final_model_path = os.path.join(checkpoint_dir, 'final_vae_model.pt') # Save in checkpoint dir
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final VAE model saved to {final_model_path}")

    if use_vertex_ai:
        # Log final metrics maybe?
        aiplatform.end_run()
        logger.info("Vertex AI run ended.")

    logger.info("VAE training finished.")


def train_vae_epoch(model, loader, optimizer, device, beta, kld_lambda):
    """Trains the GraphVAE for one epoch."""
    model.train() # Set model to training mode
    
    # Important for LoRA: Ensure only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_original_kld_loss = 0 # Track original KLD
    
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # --- Fix: Unpack 5 values from model forward pass ---
        recon_nodes, recon_adj_logits, recon_bond_logits, mu, log_var = model(batch) 
        # ---------------------------------------------------
        loss, loss_components = GraphVAE.loss_function(
            recon_nodes, recon_adj_logits, recon_bond_logits, batch, mu, log_var, beta=beta,
            kld_free_bits_lambda=kld_lambda # Pass lambda here
        )
        
        if torch.isnan(loss):
            logger.warning(f"NaN loss encountered in batch {batch_idx}. Skipping batch.")
            continue # Skip this batch

        loss.backward()
        
        # Check gradients before optimizer step (optional debugging)
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             logger.warning(f"NaN gradient detected for parameter {name} in batch {batch_idx}")
        #         # Optionally zero out NaN gradients if needed, though skipping is safer

        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_recon_loss += loss_components['recon_loss'] # item() applied in loss_fn
        total_kld_loss += loss_components['kld_loss'] # item() applied in loss_fn
        total_original_kld_loss += loss_components['original_kld_loss'] # Track original KLD
        
        if (batch_idx + 1) % 10 == 0: # Log progress every 10 batches
             elapsed_time = time.time() - start_time
             logger.info(f'  Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s')
             start_time = time.time() # Reset timer

    num_samples = len(loader.dataset)
    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kld_loss = total_kld_loss / num_samples
    avg_original_kld_loss = total_original_kld_loss / num_samples # Calculate average original KLD
    
    return avg_loss, avg_recon_loss, avg_kld_loss, avg_original_kld_loss # Return original KLD too

def evaluate_vae(model, loader, device, beta, kld_lambda):
    """Evaluates the GraphVAE on a dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_original_kld_loss = 0 # Track original KLD
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # --- Fix: Unpack 5 values from model forward pass ---
            recon_nodes, recon_adj_logits, recon_bond_logits, mu, log_var = model(batch)
            # ---------------------------------------------------
            loss, loss_components = GraphVAE.loss_function(
                recon_nodes, recon_adj_logits, recon_bond_logits, batch, mu, log_var, beta=beta,
                kld_free_bits_lambda=kld_lambda # Pass lambda here
            )
            
            if torch.isnan(loss):
                logger.warning(f"NaN loss encountered during evaluation. Skipping batch.")
                continue # Skip this batch
            
            total_loss += loss.item() * batch.num_graphs
            total_recon_loss += loss_components['recon_loss']
            total_kld_loss += loss_components['kld_loss']
            total_original_kld_loss += loss_components['original_kld_loss'] # Track original KLD

    num_samples = len(loader.dataset)
    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0 # Avoid division by zero

    avg_loss = total_loss / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kld_loss = total_kld_loss / num_samples
    avg_original_kld_loss = total_original_kld_loss / num_samples # Calculate average original KLD

    return avg_loss, avg_recon_loss, avg_kld_loss, avg_original_kld_loss # Return original KLD too

def main():
    logger.info("Starting VAE Training Script...")
    # --- Configuration --- 
    config_path = 'config/training_config.json'
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    vae_model_cfg = config['vae_model_config']
    vae_train_cfg = config['vae_training_config']
    data_cfg = config['data_config'] 
    # prop_model_cfg = config['model_config'] # Needed for node_features_dim
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Data --- 
    processed_data_dir = 'data/processed'
    logger.info(f"Loading processed data from {processed_data_dir}")
    try:
        # Use the utility function - CORRECTED FILENAMES
        train_dataset = load_processed_data(os.path.join(processed_data_dir, 'train.pt'))
        val_dataset = load_processed_data(os.path.join(processed_data_dir, 'val.pt'))
        # test_dataset = load_processed_data(os.path.join(processed_data_dir, 'test.pt')) # Optional - corrected name
        logger.info(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data: {e}. Ensure train.pt/val.pt exist. Did you run preprocess_data.py?")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading data: {e}")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=vae_train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=vae_train_cfg['batch_size'], shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=vae_train_cfg['batch_size'], shuffle=False)

    # Run the main training function - CORRECTED
    # The config contains all necessary info; train_loader etc. are created inside train_vae
    train_vae(config) 


if __name__ == "__main__":
    main() 
