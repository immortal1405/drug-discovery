"""
Dataset factory for creating different types of molecular datasets.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .datasets import (
    ZINCDataset,
    ChEMBLDataset,
    MoleculeNetDataset,
    CustomDataset
)

logger = logging.getLogger(__name__)

class DatasetFactory:
    """Factory class for creating molecular datasets."""
    
    _dataset_registry: Dict[str, type] = {
        'zinc': ZINCDataset,
        'chembl': ChEMBLDataset,
        'moleculenet': MoleculeNetDataset,
        'custom': CustomDataset
    }
    
    @classmethod
    def create_dataset(
        cls,
        dataset_type: str,
        data_path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Create a dataset instance.
        
        Args:
            dataset_type: Type of dataset to create
            data_path: Path to the dataset
            **kwargs: Additional arguments for dataset initialization
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset type is not registered
        """
        if dataset_type not in cls._dataset_registry:
            raise ValueError(
                f"Dataset type '{dataset_type}' not registered. "
                f"Available types: {list(cls._dataset_registry.keys())}"
            )
        
        dataset_class = cls._dataset_registry[dataset_type]
        logger.info(f"Creating {dataset_type} dataset from {data_path}")
        
        return dataset_class(data_path, **kwargs)
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: type) -> None:
        """
        Register a new dataset type.
        
        Args:
            name: Name of the dataset type
            dataset_class: Dataset class to register
        """
        cls._dataset_registry[name] = dataset_class
        logger.info(f"Registered new dataset type: {name}")
    
    @classmethod
    def get_available_datasets(cls) -> list:
        """
        Get list of available dataset types.
        
        Returns:
            List of dataset type names
        """
        return list(cls._dataset_registry.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_type: str) -> Dict[str, Any]:
        """
        Get information about a dataset type.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Dictionary containing dataset information
            
        Raises:
            ValueError: If dataset type is not registered
        """
        if dataset_type not in cls._dataset_registry:
            raise ValueError(f"Dataset type '{dataset_type}' not registered")
        
        dataset_class = cls._dataset_registry[dataset_type]
        return {
            'name': dataset_type,
            'class': dataset_class.__name__,
            'docstring': dataset_class.__doc__,
            'parameters': {
                name: param.annotation
                for name, param in dataset_class.__init__.__annotations__.items()
                if name != 'return'
            }
        } 