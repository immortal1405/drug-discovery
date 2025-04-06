"""
Data pipeline module for handling the complete data processing workflow.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import MolecularDataset
from .datasets import (
    ZINCDataset,
    ChEMBLDataset,
    MoleculeNetDataset,
    CustomDataset
)
from .preprocessing import MolecularPreprocessor
from .dataset_factory import DatasetFactory

logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline for handling molecular data processing workflow."""
    
    def __init__(
        self,
        dataset_type: str,
        data_path: Union[str, Path],
        preprocessor: Optional[MolecularPreprocessor] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize data pipeline.
        
        Args:
            dataset_type: Type of dataset to use
            data_path: Path to the dataset
            preprocessor: Optional preprocessor instance
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            train_split: Training set split ratio
            val_split: Validation set split ratio
            test_split: Test set split ratio
            random_seed: Random seed for reproducibility
            **kwargs: Additional arguments for dataset creation
        """
        self.dataset_type = dataset_type
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Initialize preprocessor if not provided
        self.preprocessor = preprocessor or MolecularPreprocessor()
        
        # Create dataset
        self.dataset = DatasetFactory.create_dataset(
            dataset_type,
            data_path,
            **kwargs
        )
        
        # Split dataset
        self._split_dataset()
        
        # Create dataloaders
        self._create_dataloaders()
    
    def _split_dataset(self) -> None:
        """Split dataset into train, validation, and test sets."""
        total_size = len(self.dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        logger.info(
            f"Dataset split: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )
    
    def _create_dataloaders(self) -> None:
        """Create dataloaders for train, validation, and test sets."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def preprocess_data(self) -> None:
        """Preprocess the entire dataset."""
        # Get all molecules from the dataset
        all_molecules = [item['molecule'] for item in self.dataset]
        
        # Fit preprocessor on training data
        train_molecules = [item['molecule'] for item in self.train_dataset]
        self.preprocessor.fit(train_molecules)
        
        # Transform all molecules
        processed_data = self.preprocessor.transform(all_molecules)
        
        # Update dataset with processed features
        for i, item in enumerate(self.dataset):
            item.update({
                'fingerprint': processed_data['fingerprints'][i],
                'descriptors': processed_data['descriptors'][i],
                'coordinates': processed_data.get('coordinates', None)
            })
    
    def get_batch(self, split: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Get a batch of data.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Dictionary of batch tensors
        """
        if split == 'train':
            loader = self.train_loader
        elif split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")
        
        batch = next(iter(loader))
        return {
            'fingerprints': batch['fingerprint'],
            'descriptors': batch['descriptors'],
            'coordinates': batch.get('coordinates'),
            'targets': batch.get('targets')
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'type': self.dataset_type,
            'total_size': len(self.dataset),
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'batch_size': self.batch_size,
            'feature_dims': {
                'fingerprint': self.preprocessor.fingerprint_size,
                'descriptors': len(self.preprocessor.compute_descriptors(
                    self.dataset[0]['molecule']
                ))
            }
        }
    
    def save_preprocessor(self, path: Union[str, Path]) -> None:
        """
        Save preprocessor state.
        
        Args:
            path: Path to save preprocessor state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'feature_stats': self.preprocessor.feature_stats,
            'fingerprint_size': self.preprocessor.fingerprint_size,
            'radius': self.preprocessor.radius,
            'use_chirality': self.preprocessor.use_chirality,
            'normalize': self.preprocessor.normalize
        }, path)
        
        logger.info(f"Saved preprocessor state to {path}")
    
    def load_preprocessor(self, path: Union[str, Path]) -> None:
        """
        Load preprocessor state.
        
        Args:
            path: Path to preprocessor state
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor state not found: {path}")
        
        state = torch.load(path)
        self.preprocessor.feature_stats = state['feature_stats']
        self.preprocessor.fingerprint_size = state['fingerprint_size']
        self.preprocessor.radius = state['radius']
        self.preprocessor.use_chirality = state['use_chirality']
        self.preprocessor.normalize = state['normalize']
        
        logger.info(f"Loaded preprocessor state from {path}")
    
    def get_property_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Get property ranges for the dataset.
        
        Returns:
            Dictionary of property ranges
        """
        return self.dataset.get_property_ranges()
    
    def get_property_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get property statistics for the dataset.
        
        Returns:
            Dictionary of property statistics
        """
        return self.dataset.get_property_stats() 