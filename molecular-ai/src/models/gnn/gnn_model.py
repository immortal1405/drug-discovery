"""
Graph Neural Network (GNN) model for molecular generation.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from ..common.base_model import BaseModel

class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for molecular graphs."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the GNN encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, dropout=dropout))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN encoder.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x

class GNNDecoder(nn.Module):
    """Graph Neural Network decoder for molecular generation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the GNN decoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, dropout=dropout))
        
        self.convs.append(GCNConv(hidden_dim, output_dim, dropout=dropout))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN decoder.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Generated node features
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x

class GNN(BaseModel):
    """Graph Neural Network for molecular generation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            output_dim: Dimension of output features
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = GNNDecoder(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph to latent space.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Tuple of mean and log variance tensors
        """
        x = self.encoder(x, edge_index)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick.
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to graph.
        
        Args:
            z: Latent vector
            edge_index: Edge indices
            
        Returns:
            Generated node features
        """
        return self.decoder(z, edge_index)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GNN.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Dictionary containing model outputs
        """
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, edge_index)
        
        # Compute loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'recon': x_recon,
            'z': z,
            'mu': mu,
            'log_var': log_var
        }
    
    def generate(
        self,
        num_nodes: int,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate new molecular graph.
        
        Args:
            num_nodes: Number of nodes in the graph
            edge_index: Edge indices
            
        Returns:
            Generated node features
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_nodes, self.latent_dim)
            return self.decode(z, edge_index)
    
    def get_latent_representation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get latent representation of graph.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Latent representation
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x, edge_index)
            return mu 