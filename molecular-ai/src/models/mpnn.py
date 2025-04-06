import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import List, Tuple, Optional

class MolecularMPNN(nn.Module):
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, num_layers: int, num_tasks: int, dropout_p: float = 0.2, heads: int = 4):
        """Initialize the MPNN model using GAT layers.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features (Note: Not directly used by GATConv layers here)
            hidden_dim: Hidden dimension size
            num_layers: Number of GAT layers
            num_tasks: Number of prediction tasks
            dropout_p: Dropout probability
            heads: Number of attention heads in GAT layers
        """
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        # Removed edge_encoder as GATConv doesn't typically use separate edge features
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        # Input layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout_p)
        )
        # Hidden layers
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout_p)
            )
        
        # Output layers for different tasks (input dimension is still hidden_dim)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, data) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass of the model using GAT layers.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tuple containing:
                - Tensor of predictions for each task, shape [batch_size, num_tasks]
                - List of attention tuples [(edge_index, attention_weights), ...] from each GAT layer
        """
        # Encode node features
        x = self.node_encoder(data.x)
        # edge_attr is not used by GATConv layers in this configuration
        
        # Store attention weights from each layer
        attention_weights_list = [] 
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            # Pass return_attention_weights=True to the forward call
            x, attention_tuple = gat_layer(x, data.edge_index, return_attention_weights=True) 
            attention_weights_list.append(attention_tuple) # Store the attention tuple
            x = F.relu(x) # Apply activation after each GAT layer
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)
        
        # Task-specific predictions
        predictions = [head(x) for head in self.task_heads]
        
        # Concatenate predictions into a single tensor
        predictions_tensor = torch.cat(predictions, dim=1)
        
        # Return predictions and the list of attention weights
        return predictions_tensor, attention_weights_list 