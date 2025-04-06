"""
Base model class for molecular generation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base class for all molecular generation models."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize the base model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            output_dim: Dimension of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Common layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate new samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model state.
        
        Args:
            path: Path to save model
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model state.
        
        Args:
            path: Path to load model from
        """
        self.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Dictionary containing model summary
        """
        return {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "total_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 