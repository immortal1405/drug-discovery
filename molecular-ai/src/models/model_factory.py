"""
Model factory for creating and managing molecular generation models.
"""

from typing import Dict, Any, Optional, Type
import torch
import logging
from .vae.vae_model import VAE
from .gan.gan_model import GAN
from .gnn.gnn_model import GNN
from .common.base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating molecular generation models."""
    
    _model_registry: Dict[str, Type[BaseModel]] = {
        'vae': VAE,
        'gan': GAN,
        'gnn': GNN
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model class.
        
        Args:
            name: Name of the model
            model_class: Model class to register
        """
        cls._model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create ('vae', 'gan', or 'gnn')
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            output_dim: Dimension of output features
            device: Device to place model on (CPU/GPU)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Model type '{model_type}' not registered. Available types: {list(cls._model_registry.keys())}")
        
        model_class = cls._model_registry[model_type]
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            **kwargs
        )
        
        if device is not None:
            model = model.to(device)
        
        logger.info(f"Created {model_type} model with {model.get_model_summary()['total_params']} parameters")
        return model
    
    @classmethod
    def load_model(
        cls,
        model_type: str,
        model_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> BaseModel:
        """
        Load a saved model.
        
        Args:
            model_type: Type of model to load
            model_path: Path to saved model
            device: Device to place model on (CPU/GPU)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Loaded model instance
        """
        model = cls.create_model(model_type, device=device, **kwargs)
        model.load(model_path)
        logger.info(f"Loaded {model_type} model from {model_path}")
        return model
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """
        Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Model type '{model_type}' not registered")
        
        model_class = cls._model_registry[model_type]
        return {
            'name': model_type,
            'class': model_class.__name__,
            'description': model_class.__doc__,
            'parameters': model_class.__init__.__doc__
        } 