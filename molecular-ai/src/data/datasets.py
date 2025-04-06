"""
Specific dataset implementations for different molecular data sources.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
from .dataset import MolecularDataset

logger = logging.getLogger(__name__)

class ZINCDataset(MolecularDataset):
    """Dataset class for ZINC database."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        subset: str = '250k',
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize ZINC dataset.
        
        Args:
            data_path: Path to ZINC data
            subset: ZINC subset ('250k', '1M', 'full')
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments
        """
        self.subset = subset
        super().__init__(data_path, split=split, **kwargs)
    
    def _load_data(self) -> None:
        """Load ZINC data."""
        if self.subset == '250k':
            file_pattern = 'zinc_250k_{}.csv'
        elif self.subset == '1M':
            file_pattern = 'zinc_1M_{}.csv'
        else:
            file_pattern = 'zinc_full_{}.csv'
        
        file_path = self.data_path / file_pattern.format(self.split)
        if not file_path.exists():
            raise FileNotFoundError(f"ZINC data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            mol_dict = self._process_molecule(row)
            if mol_dict:
                self.molecules.append(mol_dict)

class ChEMBLDataset(MolecularDataset):
    """Dataset class for ChEMBL database."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        target_type: str = 'IC50',
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize ChEMBL dataset.
        
        Args:
            data_path: Path to ChEMBL data
            target_type: Type of target measurement
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments
        """
        self.target_type = target_type
        super().__init__(data_path, split=split, **kwargs)
    
    def _load_data(self) -> None:
        """Load ChEMBL data."""
        file_path = self.data_path / f'chembl_{self.target_type}_{self.split}.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"ChEMBL data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            mol_dict = self._process_molecule(row)
            if mol_dict:
                self.molecules.append(mol_dict)

class MoleculeNetDataset(MolecularDataset):
    """Dataset class for MoleculeNet benchmark."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        task: str = 'ESOL',
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize MoleculeNet dataset.
        
        Args:
            data_path: Path to MoleculeNet data
            task: MoleculeNet task name
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments
        """
        self.task = task
        super().__init__(data_path, split=split, **kwargs)
    
    def _load_data(self) -> None:
        """Load MoleculeNet data."""
        file_path = self.data_path / f'{self.task}_{self.split}.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"MoleculeNet data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            mol_dict = self._process_molecule(row)
            if mol_dict:
                self.molecules.append(mol_dict)

class CustomDataset(MolecularDataset):
    """Dataset class for custom molecular data."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        smiles_column: str = 'smiles',
        property_columns: Optional[List[str]] = None,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize custom dataset.
        
        Args:
            data_path: Path to custom data
            smiles_column: Name of SMILES column
            property_columns: List of property column names
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments
        """
        self.smiles_column = smiles_column
        self.property_columns = property_columns or []
        super().__init__(data_path, split=split, **kwargs)
    
    def _load_data(self) -> None:
        """Load custom data."""
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.xlsx':
            df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        if self.smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{self.smiles_column}' not found in data")
        
        for _, row in df.iterrows():
            mol_dict = self._process_molecule(row)
            if mol_dict:
                self.molecules.append(mol_dict)
    
    def _process_molecule(self, data: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a molecule from custom data."""
        try:
            mol = Chem.MolFromSmiles(data[self.smiles_column])
            if mol is None:
                return None
            
            # Get properties from specified columns
            properties = {}
            for col in self.property_columns:
                if col in data:
                    properties[col] = data[col]
            
            # Compute basic properties
            mol_dict = {
                'smiles': Chem.MolToSmiles(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'qed': QED.default(mol),
                'properties': properties
            }
            
            # Add target properties
            for prop in self.target_properties:
                if prop in data:
                    mol_dict[prop] = data[prop]
                else:
                    # Compute RDKit properties if not in data
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