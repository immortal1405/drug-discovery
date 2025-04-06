import json
import logging
import os
import sys # Import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from pathlib import Path
from typing import Dict, List
from google.cloud import aiplatform
from datetime import datetime

from src.models.mpnn import MolecularMPNN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config['project_id'], 
            location=config['location'], 
            experiment=config['experiment_name']
        )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load datasets and get normalization stats
        self.train_loader, self.val_loader, self.test_loader, self.y_mean, self.y_std = self.load_and_normalize_data()
        
        # Determine feature sizes from data
        sample_data = next(iter(self.train_loader))
        num_node_features = sample_data.num_node_features
        num_edge_features = sample_data.num_edge_features
        logger.info(f"Detected node features: {num_node_features}")
        logger.info(f"Detected edge features: {num_edge_features}")
        
        # Create model (pass dropout_p if defined in config, else default)
        dropout_p = config.get('model_config', {}).get('dropout_p', 0.2) 
        self.model = MolecularMPNN(
            node_features=num_node_features,
            edge_features=num_edge_features,
            hidden_dim=config['model_config']['hidden_dim'],
            num_layers=config['model_config']['num_layers'],
            num_tasks=config['model_config']['num_tasks'],
            dropout_p=dropout_p
        ).to(self.device)

        # Define optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training_config']['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Add Learning Rate Scheduler
        scheduler_patience = config.get('training_config', {}).get('scheduler_patience', 5)
        scheduler_factor = config.get('training_config', {}).get('scheduler_factor', 0.5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience, 
            verbose=True
        )
        
        # Early stopping parameters
        self.patience = config['training_config']['patience']
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def load_and_normalize_data(self):
        # Load processed data
        train_dataset = torch.load(os.path.join('data', 'processed', 'train.pt'), weights_only=False)
        val_dataset = torch.load(os.path.join('data', 'processed', 'val.pt'), weights_only=False)
        test_dataset = torch.load(os.path.join('data', 'processed', 'test.pt'), weights_only=False)

        # Calculate mean and std from training set targets
        all_train_y = torch.cat([data.y.unsqueeze(0) for data in train_dataset], dim=0)
        y_mean = all_train_y.mean(dim=0)
        y_std = all_train_y.std(dim=0)
        # Add small epsilon to std to avoid division by zero for constant targets
        y_std = torch.where(y_std == 0, torch.tensor(1.0), y_std) 
        
        logger.info(f"Target Mean: {y_mean}")
        logger.info(f"Target Std: {y_std}")

        # Normalize targets
        for data in train_dataset:
            data.y = (data.y - y_mean) / y_std
        for data in val_dataset:
            data.y = (data.y - y_mean) / y_std
        for data in test_dataset:
            data.y = (data.y - y_mean) / y_std
            
        # Move mean and std to the correct device
        y_mean = y_mean.to(self.device)
        y_std = y_std.to(self.device)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['training_config']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['training_config']['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['training_config']['batch_size'])
        
        return train_loader, val_loader, test_loader, y_mean, y_std
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        task_losses = [0] * self.config['model_config']['num_tasks']
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass - returns ([batch_size, num_tasks], attention_weights_list)
            predictions_norm, _ = self.model(batch) # Unpack the tuple
            
            # Targets are now correctly shaped [batch_size, num_tasks]
            targets_norm = batch.y

            # Compute losses - refactored for NaN handling
            # Mask for valid (non-NaN) targets
            valid_mask = ~torch.isnan(targets_norm)

            # Filter predictions and targets using the mask with torch.masked_select
            predictions_valid = torch.masked_select(predictions_norm, valid_mask)
            targets_valid = torch.masked_select(targets_norm, valid_mask)

            # Compute MSE loss on valid pairs only
            if predictions_valid.numel() > 0: # Check if there are any valid targets
                loss = self.criterion(predictions_valid, targets_valid)
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Handle cases with all NaNs

            total_loss += loss.item() * batch.num_graphs # Weight by batch size

            # Calculate and store task-specific losses (optional, for logging)
            with torch.no_grad(): # No gradients needed for logging losses
                 for i in range(self.config['model_config']['num_tasks']):
                     task_mask_batch = valid_mask[:, i]
                     if task_mask_batch.any(): # Check if any valid targets for this task in batch
                         # Use masked_select for task-specific loss calculation too
                         pred_task_valid = torch.masked_select(predictions_norm[:, i], task_mask_batch)
                         target_task_valid = torch.masked_select(targets_norm[:, i], task_mask_batch)
                         task_loss_batch = self.criterion(pred_task_valid, target_task_valid)
                         task_losses[i] += task_loss_batch.item() * batch.num_graphs
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        # Average losses
        num_samples = len(self.train_loader.dataset)
        metrics = {
            'train_loss': total_loss / num_samples,
            **{f'train_task_{i}_loss': task_loss_sum / num_samples 
               for i, task_loss_sum in enumerate(task_losses)}
        }
        
        return metrics
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Dictionary of evaluation metrics (loss, MAE, RMSE per task)
        """
        self.model.eval()
        total_loss = 0
        task_losses = [0.0] * self.config['model_config']['num_tasks']
        # Initialize lists to store batch-wise summed errors as tensors
        task_mae_tensors = [[] for _ in range(self.config['model_config']['num_tasks'])]
        task_mse_tensors = [[] for _ in range(self.config['model_config']['num_tasks'])]
        num_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                current_batch_size = batch.num_graphs
                num_samples += current_batch_size
                
                # Forward pass (predictions are normalized, graph-level)
                # Returns ([batch_size, num_tasks], attention_weights_list)
                predictions_norm, _ = self.model(batch) # Unpack the tuple

                # Targets are now correctly shaped [batch_size, num_tasks]
                targets_norm = batch.y
                
                # Compute normalized loss
                losses = []
                # Compare graph-level predictions and targets
                for i in range(self.config['model_config']['num_tasks']):
                    loss = self.criterion(predictions_norm[:, i], targets_norm[:, i])
                    losses.append(loss)
                    task_losses[i] += loss.item() * current_batch_size # Weight loss by batch size
                
                # Un-normalize for MAE/RMSE calculation
                # Ensure y_mean and y_std are correctly shaped for broadcasting
                # Reshape mean/std to [1, num_tasks] for broadcasting with [batch_size, num_tasks]
                current_y_mean = self.y_mean.unsqueeze(0) # Shape [1, num_tasks]
                current_y_std = self.y_std.unsqueeze(0)   # Shape [1, num_tasks]

                targets_unnorm = targets_norm * current_y_std + current_y_mean
                # Apply un-normalization to the correct tensor
                predictions_unnorm = predictions_norm * current_y_std + current_y_mean

                # Compute MAE and MSE sums for the batch
                mae_batch = torch.abs(predictions_unnorm - targets_unnorm).sum(dim=0) # Shape [num_tasks]
                mse_batch = torch.pow(predictions_unnorm - targets_unnorm, 2).sum(dim=0) # Shape [num_tasks]
                
                # Append the summed error tensor for each task
                for i in range(self.config['model_config']['num_tasks']):
                    task_mae_tensors[i].append(mae_batch[i]) # Append 0-dim tensor
                    task_mse_tensors[i].append(mse_batch[i]) # Append 0-dim tensor
                
                # Total loss (normalized)
                total_loss += sum(losses).item() * current_batch_size
        
        # Average losses and calculate final metrics
        metrics = {
            'loss': total_loss / num_samples # Average normalized loss
        }
        for i in range(self.config['model_config']['num_tasks']):
            # Sum the tensors for each task, convert to float, then divide
            total_task_mae = torch.stack(task_mae_tensors[i]).sum().item()
            total_task_mse = torch.stack(task_mse_tensors[i]).sum().item()
            
            metrics[f'task_{i}_loss'] = task_losses[i] / num_samples
            metrics[f'task_{i}_mae'] = total_task_mae / num_samples
            metrics[f'task_{i}_rmse'] = (total_task_mse / num_samples) ** 0.5

        return metrics
    
    def train(self):
        logger.info("Starting training...")
        num_epochs = self.config['training_config']['num_epochs']
        
        # Create Vertex AI experiment run
        with aiplatform.start_run(run=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log hyperparameters
            params_to_log = {
                'learning_rate': self.config['training_config']['learning_rate'],
                'patience': self.config['training_config']['patience'],
                'batch_size': self.config['training_config']['batch_size'],
                'num_epochs': self.config['training_config']['num_epochs'],
                'model_hidden_dim': self.config['model_config']['hidden_dim'],
                'model_num_layers': self.config['model_config']['num_layers'],
                'model_num_tasks': self.config['model_config']['num_tasks'],
                # Add other relevant config values as needed
            }
            aiplatform.log_params(params_to_log)
            
            for epoch in range(num_epochs):
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Evaluate
                val_metrics = self.evaluate(self.val_loader)
                
                # Log metrics
                metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                aiplatform.log_metrics(metrics)
                
                # Log epoch
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Val Task 0 MAE: {val_metrics['task_0_mae']:.4f}") # Example: log MAE for task 0
                
                # Step the LR scheduler based on validation loss
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_no_improve = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'models/best_model.pt')
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        logger.info("Early stopping triggered")
                        break
            
            # Evaluate on test set after training completes
            logger.info("Evaluating on test set...")
            test_metrics = self.evaluate(self.test_loader)
            logger.info(f"Test Results: {test_metrics}")
            
            # Log test metrics
            aiplatform.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Save final model
            torch.save(self.model.state_dict(), 'models/final_model.pt')
            logger.info("Training finished. Best model saved to models/best_model.pt, final model saved to models/final_model.pt")

def main():
    """Main function to run the training pipeline."""
    # Load configuration
    with open('config/training_config.json', 'r') as f:
        config = json.load(f)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.train()

if __name__ == "__main__":
    main() 