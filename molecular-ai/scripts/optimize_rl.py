# molecular-ai/scripts/optimize_rl.py

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import time

import torch
import torch.optim as optim
from torch.distributions import Categorical
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.graph_vae import GraphVAE
from src.models.mpnn import MolecularMPNN # Assuming this is the property predictor class
from src.utils.data_utils import smiles_to_graph_data # Function to convert SMILES to graph format

# --- Logging Setup ---
log_file = Path("logs") / f"optimize_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


# --- Helper Functions ---

def calculate_reward(smiles_list: List[str], predictor_model: MolecularMPNN, device: torch.device, target_logp: float = 3.0, validity_weight: float = 1.0, property_weight: float = 1.0) -> Tuple[torch.Tensor, Dict]:
    """Calculates reward for a batch of generated SMILES strings."""
    rewards = []
    valid_count = 0
    property_scores = []

    predictor_model.eval() # Set predictor to evaluation mode
    with torch.no_grad():
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            reward = 0.0
            
            # 1. Validity Reward/Penalty
            if mol is not None:
                reward += validity_weight 
                valid_count += 1
                
                # 2. Property Reward (e.g., LogP close to target)
                try:
                    graph_data = smiles_to_graph_data(smiles) # Convert valid SMILES to graph
                    if graph_data:
                        graph_data = graph_data.to(device)
                        # Assuming predictor takes a Data object or batch
                        predicted_props = predictor_model(graph_data.x, graph_data.edge_index, graph_data.batch if hasattr(graph_data, 'batch') else torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device))
                        predicted_logp = predicted_props[0].item() # Assuming LogP is the first output
                        property_scores.append(predicted_logp)
                        
                        # Example: Gaussian reward around target_logp
                        logp_reward = max(0, 1.0 - abs(predicted_logp - target_logp) / 2.0) # Simple linear penalty 
                        reward += property_weight * logp_reward
                    else:
                         logger.warning(f"Could not convert valid SMILES to graph: {smiles}")
                         # Penalize if conversion fails? Or just give validity reward?
                         
                except Exception as e:
                     logger.error(f"Error during property prediction for {smiles}: {e}")
                     # Penalize property prediction errors?
                     
            else: # Invalid molecule
                reward -= validity_weight # Simple penalty for invalidity 
                property_scores.append(float('nan')) # Indicate invalid property
                
            rewards.append(reward)
            
    avg_property = sum(p for p in property_scores if not torch.isnan(torch.tensor(p))) / valid_count if valid_count > 0 else 0
    validity = valid_count / len(smiles_list) if smiles_list else 0
    
    metrics = {
        'validity': validity,
        'avg_logp': avg_property
    }
    return torch.tensor(rewards, dtype=torch.float32, device=device), metrics

# --- RL Optimization Pipeline ---

def optimize_generator_rl(config: Dict):
    """Fine-tunes the Graph VAE generator using RL."""
    logger.info("Starting RL optimization pipeline...")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    vae_config = config['vae_model_config']
    predictor_config = config['model_config'] # Assuming property predictor config is here
    rl_config = config['rl_optimization_config']
    
    # Checkpoint paths from config
    vae_checkpoint_path = rl_config['vae_checkpoint_path']
    predictor_checkpoint_path = rl_config['predictor_checkpoint_path']
    output_dir = Path(rl_config.get('output_dir', 'models/rl_optimized_vae'))
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / "rl_optimized_vae_model.pt"

    # --- Load Models ---
    logger.info(f"Loading pre-trained VAE from: {vae_checkpoint_path}")
    # Need node_features dim - assuming it's in predictor_config or derivable
    # This might need adjustment based on actual config structure
    node_features_dim = predictor_config.get('node_features', 120) 
    if not os.path.exists(vae_checkpoint_path):
         logger.error(f"VAE checkpoint not found: {vae_checkpoint_path}")
         return
    vae_model = GraphVAE(
        node_features=node_features_dim, # Needs to match the trained VAE
        hidden_dim=vae_config['hidden_dim'],
        latent_dim=vae_config['latent_dim'],
        max_nodes=vae_config['max_nodes'], 
        num_enc_layers=vae_config['num_enc_layers'], heads_enc=vae_config['heads_enc'], dropout_enc=vae_config['dropout_enc'],
        num_dec_layers=vae_config['num_dec_layers'], heads_dec=vae_config['heads_dec'], dropout_dec=vae_config['dropout_dec'],
        lora_r=vae_config.get('lora_r', 0), lora_alpha=vae_config.get('lora_alpha', 1), lora_dropout=vae_config.get('lora_dropout', 0)
    ).to(device)
    vae_model.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    logger.info("VAE model loaded successfully.")
    
    logger.info(f"Loading property predictor from: {predictor_checkpoint_path}")
    if not os.path.exists(predictor_checkpoint_path):
         logger.error(f"Property predictor checkpoint not found: {predictor_checkpoint_path}")
         return
    predictor_model = MolecularMPNN(
        node_features=node_features_dim, # Should match predictor training
        edge_features=predictor_config.get('edge_features', 10), # Example value
        hidden_dim=predictor_config['hidden_dim'],
        output_dim=predictor_config['output_dim'], # e.g., 1 if predicting only LogP
        num_layers=predictor_config['num_layers'],
        dropout=predictor_config['dropout']
    ).to(device)
    predictor_model.load_state_dict(torch.load(predictor_checkpoint_path, map_location=device))
    predictor_model.eval() # Predictor is not trained further
    logger.info("Property predictor loaded successfully.")

    # --- Optimizer (for VAE Decoder/Generator) ---
    # Decide which parts of VAE to fine-tune (e.g., only decoder?)
    # For simplicity, let's try fine-tuning the whole VAE initially
    optimizer = optim.Adam(
        vae_model.parameters(), 
        lr=rl_config.get('learning_rate', 1e-4)
    )
    logger.info(f"Optimizer: Adam, LR: {rl_config.get('learning_rate', 1e-4)}")

    # --- RL Training Loop ---
    num_iterations = rl_config.get('num_iterations', 1000)
    batch_size = rl_config.get('batch_size', 32)
    target_logp = rl_config.get('target_logp', 3.0)
    validity_weight = rl_config.get('validity_weight', 1.0)
    property_weight = rl_config.get('property_weight', 1.0)
    
    # Simple REINFORCE-like update
    log_probs_history = []
    rewards_history = []
    baseline = torch.tensor(0.0, device=device) # Simple moving average baseline
    baseline_alpha = 0.9 # Smoothing factor for baseline

    logger.info(f"Starting RL optimization for {num_iterations} iterations...")

    vae_model.train() # Keep VAE in train mode for fine-tuning

    for iteration in range(1, num_iterations + 1):
        iteration_start_time = time.time()
        
        # 1. Sample molecules from the VAE generator (decoder)
        # This requires a function in GraphVAE to sample from latent space
        # Let's assume vae_model.sample(num_samples) returns SMILES list 
        # This `sample` method needs to be implemented in GraphVAE!
        try:
            with torch.no_grad(): # Sampling doesn't require gradients initially
                 # Assuming latent_dim is accessible or known
                 z = torch.randn(batch_size, vae_model.latent_dim).to(device)
                 # Assuming decode returns graph components, need conversion to SMILES
                 # This part is complex and depends heavily on VAE implementation
                 # Placeholder: Replace with actual sampling and SMILES conversion
                 generated_smiles: List[str] = vae_model.sample_smiles(z) 
                 if not generated_smiles:
                     logger.warning(f"Iteration {iteration}: VAE sampling returned no SMILES.")
                     continue
        except NotImplementedError:
            logger.error("GraphVAE needs a `sample_smiles(z)` method implemented to generate SMILES from latent vectors for RL.")
            return
        except Exception as e:
            logger.error(f"Error during VAE sampling: {e}")
            continue

        # 2. Calculate Rewards
        rewards, metrics = calculate_reward(
            generated_smiles, predictor_model, device, target_logp, validity_weight, property_weight
        )
        
        # 3. Calculate Loss (Policy Gradient - REINFORCE)
        # This step requires the VAE's decode process to output log probabilities 
        # of the generation steps (actions) to calculate the policy gradient.
        # This is a MAJOR simplification. A proper implementation is complex.
        
        # Placeholder: We need log_prob of generated SMILES from VAE
        # log_probs = vae_model.get_log_probs(generated_smiles) # Needs implementation!
        # For now, let's use a dummy log_prob calculation for structure
        dummy_log_probs = torch.randn(batch_size, requires_grad=True).to(device) # Replace with real log_probs
        
        # Update baseline (moving average of rewards)
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * rewards.mean()
        
        # Calculate Advantage
        advantages = rewards - baseline.detach() # Detach baseline to treat it as constant
        
        # Policy Loss: - (log_prob * advantage)
        # Summing over batch - needs adjustment based on actual log_prob structure
        policy_loss = -(dummy_log_probs * advantages).mean() 

        # 4. Backpropagate and Update VAE
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Logging
        iteration_duration = time.time() - iteration_start_time
        if iteration % rl_config.get('log_interval', 10) == 0:
            logger.info(f"Iter {iteration}/{num_iterations} | Loss: {policy_loss.item():.4f} | Reward Mean: {rewards.mean().item():.3f} | Baseline: {baseline.item():.3f} | Validity: {metrics['validity']:.2f} | Avg LogP: {metrics['avg_logp']:.3f} | Time: {iteration_duration:.2f}s")

        # Optional: Save checkpoints periodically
        if iteration % rl_config.get('checkpoint_interval', 100) == 0:
            chkpt_path = output_dir / f"rl_vae_checkpoint_iter_{iteration}.pt"
            torch.save(vae_model.state_dict(), chkpt_path)
            logger.info(f"Saved RL VAE checkpoint to {chkpt_path}")

    # --- Save Final Optimized Model ---
    torch.save(vae_model.state_dict(), final_model_path)
    logger.info(f"Final RL optimized VAE model saved to {final_model_path}")

    logger.info("RL optimization finished.")


def main():
    """Loads config and runs the RL optimization."""
    logger.info("Starting RL Optimization Script...")
    config_path = 'config/training_config.json' # Assuming RL config is added here
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        return
        
    # --- Validate Config for RL ---
    if 'rl_optimization_config' not in config:
        logger.error("'rl_optimization_config' section missing from config.")
        return
    if 'vae_checkpoint_path' not in config['rl_optimization_config']:
        logger.error("'vae_checkpoint_path' missing from rl_optimization_config.")
        return
    if 'predictor_checkpoint_path' not in config['rl_optimization_config']:
        logger.error("'predictor_checkpoint_path' missing from rl_optimization_config.")
        return
        
    optimize_generator_rl(config)

if __name__ == "__main__":
    main() 