"""
Data downloader module for handling dataset downloads and preparation.
"""

import logging
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloader for molecular datasets."""
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False
    ):
        """
        Initialize dataset downloader.
        
        Args:
            cache_dir: Directory to cache downloaded files
            force_download: Whether to force redownload
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.molecular_ai' / 'datasets'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_download = force_download
        
        # Dataset URLs and information
        self.dataset_info = {
            'zinc': {
                'urls': {
                    '250k': 'https://zinc.docking.org/db/bysubset/250k/250k.csv',
                    '1M': 'https://zinc.docking.org/db/bysubset/1M/1M.csv',
                    'full': 'https://zinc.docking.org/db/bysubset/full/full.csv'
                },
                'description': 'ZINC database of commercially available compounds'
            },
            'chembl': {
                'urls': {
                    'IC50': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/chembl_IC50.csv',
                    'Ki': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/chembl_Ki.csv'
                },
                'description': 'ChEMBL database of bioactive molecules'
            },
            'moleculenet': {
                'urls': {
                    'ESOL': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/ESOL.csv',
                    'FreeSolv': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/FreeSolv.csv',
                    'Lipophilicity': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/Lipophilicity.csv'
                },
                'description': 'MoleculeNet benchmark datasets'
            }
        }
    
    def download_file(
        self,
        url: str,
        filename: str,
        desc: Optional[str] = None
    ) -> Path:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            filename: Name to save file as
            desc: Description for progress bar
            
        Returns:
            Path to downloaded file
        """
        filepath = self.cache_dir / filename
        
        if filepath.exists() and not self.force_download:
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filepath, 'wb') as f, tqdm(
            desc=desc or filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
        
        return filepath
    
    def download_dataset(
        self,
        dataset_type: str,
        subset: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Path]:
        """
        Download a dataset.
        
        Args:
            dataset_type: Type of dataset to download
            subset: Dataset subset (if applicable)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of downloaded file paths
            
        Raises:
            ValueError: If dataset type or subset is invalid
        """
        if dataset_type not in self.dataset_info:
            raise ValueError(
                f"Invalid dataset type: {dataset_type}. "
                f"Available types: {list(self.dataset_info.keys())}"
            )
        
        info = self.dataset_info[dataset_type]
        downloaded_files = {}
        
        if subset:
            if subset not in info['urls']:
                raise ValueError(
                    f"Invalid subset: {subset}. "
                    f"Available subsets: {list(info['urls'].keys())}"
                )
            urls = {subset: info['urls'][subset]}
        else:
            urls = info['urls']
        
        for name, url in urls.items():
            filename = f"{dataset_type}_{name}.csv"
            filepath = self.download_file(
                url,
                filename,
                desc=f"Downloading {dataset_type} {name}"
            )
            downloaded_files[name] = filepath
        
        return downloaded_files
    
    def prepare_dataset(
        self,
        dataset_type: str,
        subset: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download and prepare a dataset.
        
        Args:
            dataset_type: Type of dataset to prepare
            subset: Dataset subset (if applicable)
            **kwargs: Additional arguments
            
        Returns:
            Prepared dataset as DataFrame
        """
        downloaded_files = self.download_dataset(dataset_type, subset)
        
        if dataset_type == 'zinc':
            return self._prepare_zinc(downloaded_files, **kwargs)
        elif dataset_type == 'chembl':
            return self._prepare_chembl(downloaded_files, **kwargs)
        elif dataset_type == 'moleculenet':
            return self._prepare_moleculenet(downloaded_files, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _prepare_zinc(
        self,
        files: Dict[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Prepare ZINC dataset."""
        dfs = []
        for name, filepath in files.items():
            df = pd.read_csv(filepath)
            df['subset'] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def _prepare_chembl(
        self,
        files: Dict[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Prepare ChEMBL dataset."""
        dfs = []
        for name, filepath in files.items():
            df = pd.read_csv(filepath)
            df['target_type'] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def _prepare_moleculenet(
        self,
        files: Dict[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Prepare MoleculeNet dataset."""
        dfs = []
        for name, filepath in files.items():
            df = pd.read_csv(filepath)
            df['task'] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def get_dataset_info(self, dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available datasets.
        
        Args:
            dataset_type: Optional dataset type to get info for
            
        Returns:
            Dictionary of dataset information
        """
        if dataset_type:
            if dataset_type not in self.dataset_info:
                raise ValueError(f"Invalid dataset type: {dataset_type}")
            return self.dataset_info[dataset_type]
        return self.dataset_info
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Whether SMILES is valid
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def filter_valid_smiles(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Filter DataFrame to keep only valid SMILES.
        
        Args:
            df: DataFrame to filter
            smiles_column: Name of SMILES column
            
        Returns:
            Filtered DataFrame
        """
        valid_mask = df[smiles_column].apply(self.validate_smiles)
        return df[valid_mask].reset_index(drop=True)
    
    def compute_basic_properties(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Compute basic molecular properties.
        
        Args:
            df: DataFrame with SMILES
            smiles_column: Name of SMILES column
            
        Returns:
            DataFrame with additional properties
        """
        properties = []
        for smiles in tqdm(df[smiles_column], desc="Computing properties"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                props = {
                    'molecular_weight': Descriptors.ExactMolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol)
                }
                properties.append(props)
            else:
                properties.append({k: None for k in props})
        
        return pd.concat([df, pd.DataFrame(properties)], axis=1) 