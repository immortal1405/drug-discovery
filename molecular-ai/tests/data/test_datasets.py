"""
Tests for molecular dataset classes and factory.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from src.data.datasets import (
    ZINCDataset,
    ChEMBLDataset,
    MoleculeNetDataset,
    CustomDataset
)
from src.data.dataset_factory import DatasetFactory

@pytest.fixture
def sample_data_path(tmp_path) -> Path:
    """Create sample data files for testing."""
    # Create ZINC data
    zinc_data = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
        'logp': [0.5, 1.2, 2.1],
        'tpsa': [20.2, 37.3, 0.0]
    })
    zinc_data.to_csv(tmp_path / 'zinc_250k_train.csv', index=False)
    
    # Create ChEMBL data
    chembl_data = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
        'target_value': [0.5, 1.2, 2.1],
        'target_type': ['IC50', 'IC50', 'IC50']
    })
    chembl_data.to_csv(tmp_path / 'chembl_IC50_train.csv', index=False)
    
    # Create MoleculeNet data
    moleculenet_data = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
        'target': [0.5, 1.2, 2.1]
    })
    moleculenet_data.to_csv(tmp_path / 'ESOL_train.csv', index=False)
    
    return tmp_path

@pytest.fixture
def custom_data_path(tmp_path) -> Path:
    """Create custom dataset file for testing."""
    custom_data = pd.DataFrame({
        'molecule_smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
        'property1': [0.5, 1.2, 2.1],
        'property2': [10.0, 20.0, 30.0]
    })
    custom_data.to_csv(tmp_path / 'custom_data.csv', index=False)
    return tmp_path

def test_zinc_dataset(sample_data_path):
    """Test ZINC dataset creation and functionality."""
    dataset = ZINCDataset(sample_data_path, subset='250k', split='train')
    assert len(dataset) == 3
    assert isinstance(dataset[0], dict)
    assert 'smiles' in dataset[0]
    assert 'logp' in dataset[0]
    assert 'tpsa' in dataset[0]

def test_chembl_dataset(sample_data_path):
    """Test ChEMBL dataset creation and functionality."""
    dataset = ChEMBLDataset(sample_data_path, target_type='IC50', split='train')
    assert len(dataset) == 3
    assert isinstance(dataset[0], dict)
    assert 'smiles' in dataset[0]
    assert 'target_value' in dataset[0]
    assert 'target_type' in dataset[0]

def test_moleculenet_dataset(sample_data_path):
    """Test MoleculeNet dataset creation and functionality."""
    dataset = MoleculeNetDataset(sample_data_path, task='ESOL', split='train')
    assert len(dataset) == 3
    assert isinstance(dataset[0], dict)
    assert 'smiles' in dataset[0]
    assert 'target' in dataset[0]

def test_custom_dataset(custom_data_path):
    """Test custom dataset creation and functionality."""
    dataset = CustomDataset(
        custom_data_path / 'custom_data.csv',
        smiles_column='molecule_smiles',
        property_columns=['property1', 'property2']
    )
    assert len(dataset) == 3
    assert isinstance(dataset[0], dict)
    assert 'smiles' in dataset[0]
    assert 'property1' in dataset[0]['properties']
    assert 'property2' in dataset[0]['properties']

def test_dataset_factory(sample_data_path):
    """Test dataset factory functionality."""
    # Test creating different dataset types
    zinc_dataset = DatasetFactory.create_dataset('zinc', sample_data_path)
    assert isinstance(zinc_dataset, ZINCDataset)
    
    chembl_dataset = DatasetFactory.create_dataset('chembl', sample_data_path)
    assert isinstance(chembl_dataset, ChEMBLDataset)
    
    # Test getting available datasets
    available_datasets = DatasetFactory.get_available_datasets()
    assert 'zinc' in available_datasets
    assert 'chembl' in available_datasets
    
    # Test getting dataset info
    zinc_info = DatasetFactory.get_dataset_info('zinc')
    assert zinc_info['name'] == 'zinc'
    assert 'parameters' in zinc_info
    
    # Test invalid dataset type
    with pytest.raises(ValueError):
        DatasetFactory.create_dataset('invalid_type', sample_data_path)

def test_dataset_property_stats(sample_data_path):
    """Test property statistics computation."""
    dataset = ZINCDataset(sample_data_path, subset='250k', split='train')
    stats = dataset.get_property_stats()
    assert 'logp' in stats
    assert 'tpsa' in stats
    assert all(key in stats['logp'] for key in ['mean', 'std', 'min', 'max'])

def test_dataset_dataloader(sample_data_path):
    """Test DataLoader creation."""
    dataset = ZINCDataset(sample_data_path, subset='250k', split='train')
    dataloader = dataset.create_dataloader(batch_size=2, shuffle=True)
    assert len(dataloader) == 2  # 3 samples with batch_size=2
    batch = next(iter(dataloader))
    assert len(batch) == 2

def test_dataset_edge_cases(sample_data_path):
    """Test dataset edge cases."""
    # Test empty dataset
    empty_data = pd.DataFrame(columns=['smiles', 'logp', 'tpsa'])
    empty_data.to_csv(sample_data_path / 'empty_zinc.csv', index=False)
    dataset = ZINCDataset(sample_data_path, subset='250k', split='train')
    assert len(dataset) == 0
    
    # Test invalid SMILES
    invalid_data = pd.DataFrame({
        'smiles': ['invalid_smiles', 'CCO'],
        'logp': [0.5, 1.2],
        'tpsa': [20.2, 37.3]
    })
    invalid_data.to_csv(sample_data_path / 'invalid_zinc.csv', index=False)
    dataset = ZINCDataset(sample_data_path, subset='250k', split='train')
    assert len(dataset) == 1  # Only valid SMILES should be included 