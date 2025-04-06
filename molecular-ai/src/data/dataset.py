"""
Dataset handling for molecular generation models.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """Base dataset class for molecular data."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[callable] = None,
        target_properties: Optional[List[str]] = None,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize the molecular dataset.
        
        Args:
            data_path: Path to the data file or directory
            transform: Optional transform to apply to the data
            target_properties: List of target properties to include
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_properties = target_properties or []
        self.split = split
        self.molecules: List[Dict[str, Any]] = []
        self.property_stats: Dict[str, Dict[str, float]] = {}
        
        self._load_data()
        self._compute_property_stats()
    
    def _load_data(self) -> None:
        """Load molecular data from the specified path."""
        if self.data_path.suffix == '.csv':
            self._load_csv()
        elif self.data_path.suffix == '.sdf':
            self._load_sdf()
        elif self.data_path.is_dir():
            self._load_directory()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
    
    def _load_csv(self) -> None:
        """Load data from CSV file."""
        df = pd.read_csv(self.data_path)
        for _, row in df.iterrows():
            mol_dict = self._process_molecule(row)
            if mol_dict:
                self.molecules.append(mol_dict)
    
    def _load_sdf(self) -> None:
        """Load data from SDF file."""
        supplier = Chem.SDMolSupplier(str(self.data_path))
        for mol in supplier:
            if mol is not None:
                mol_dict = self._process_molecule(mol)
                if mol_dict:
                    self.molecules.append(mol_dict)
    
    def _load_directory(self) -> None:
        """Load data from directory of molecular files."""
        for file_path in self.data_path.glob('*'):
            if file_path.suffix in ['.sdf', '.mol', '.mol2']:
                mol = Chem.SDMolSupplier(str(file_path))[0]
                if mol is not None:
                    mol_dict = self._process_molecule(mol)
                    if mol_dict:
                        self.molecules.append(mol_dict)
    
    def _process_molecule(
        self,
        data: Union[pd.Series, Chem.Mol]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a molecule into a dictionary format.
        
        Args:
            data: Molecule data (either pandas Series or RDKit Mol)
            
        Returns:
            Dictionary containing processed molecule data
        """
        try:
            if isinstance(data, pd.Series):
                mol = Chem.MolFromSmiles(data['smiles'])
                properties = data.to_dict()
            else:
                mol = data
                properties = {}
            
            if mol is None:
                return None
            
            # Compute basic properties
            mol_dict = {
                'smiles': Chem.MolToSmiles(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'qed': QED.default(mol),
                'properties': properties
            }
            
            # Compute additional properties if requested
            for prop in self.target_properties:
                if prop == 'logp':
                    mol_dict['logp'] = Descriptors.MolLogP(mol)
                elif prop == 'tpsa':
                    mol_dict['tpsa'] = Descriptors.TPSA(mol)
                elif prop == 'rotatable_bonds':
                    mol_dict['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                elif prop == 'hbd':
                    mol_dict['hbd'] = Descriptors.NumHDonors(mol)
                elif prop == 'hba':
                    mol_dict['hba'] = Descriptors.NumHAcceptors(mol)
            
            return mol_dict
            
        except Exception as e:
            logger.warning(f"Error processing molecule: {e}")
            return None
    
    def _compute_property_stats(self) -> None:
        """Compute statistics for numerical properties."""
        for prop in self.target_properties:
            values = [mol[prop] for mol in self.molecules if prop in mol]
            if values:
                self.property_stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    def __len__(self) -> int:
        """Get the number of molecules in the dataset."""
        return len(self.molecules)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a molecule and its properties.
        
        Args:
            idx: Index of the molecule
            
        Returns:
            Dictionary containing molecule data and properties
        """
        mol_dict = self.molecules[idx]
        
        if self.transform:
            mol_dict = self.transform(mol_dict)
        
        return mol_dict
    
    def get_property_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for numerical properties.
        
        Returns:
            Dictionary containing property statistics
        """
        return self.property_stats
    
    def get_property_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get valid ranges for numerical properties.
        
        Returns:
            Dictionary containing property ranges
        """
        return {
            prop: (stats['min'], stats['max'])
            for prop, stats in self.property_stats.items()
        }
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        ) 