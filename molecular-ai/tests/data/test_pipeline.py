"""
Tests for molecular data pipeline.
"""

import pytest
import torch
from pathlib import Path
from typing import Dict, Any
from src.data.pipeline import DataPipeline
from src.data.preprocessing import MolecularPreprocessor

@pytest.fixture
def sample_data_path(tmp_path) -> Path:
    """Create sample data files for testing."""
    # Create ZINC data
    zinc_data = pd.DataFrame({
        'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'] * 10,  # 30 samples
        'logp': [0.5, 1.2, 2.1] * 10,
        'tpsa': [20.2, 37.3, 0.0] * 10
    })
    zinc_data.to_csv(tmp_path / 'zinc_250k_train.csv', index=False)
    return tmp_path

@pytest.fixture
def pipeline(sample_data_path):
    """Create data pipeline instance for testing."""
    return DataPipeline(
        dataset_type='zinc',
        data_path=sample_data_path,
        batch_size=4,
        num_workers=0,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )

def test_pipeline_initialization(pipeline):
    """Test pipeline initialization."""
    assert pipeline.dataset_type == 'zinc'
    assert pipeline.batch_size == 4
    assert pipeline.num_workers == 0
    assert pipeline.train_split == 0.8
    assert pipeline.val_split == 0.1
    assert pipeline.test_split == 0.1
    assert isinstance(pipeline.preprocessor, MolecularPreprocessor)

def test_dataset_splitting(pipeline):
    """Test dataset splitting."""
    total_size = len(pipeline.dataset)
    train_size = len(pipeline.train_dataset)
    val_size = len(pipeline.val_dataset)
    test_size = len(pipeline.test_dataset)
    
    assert total_size == 30
    assert train_size == 24  # 80% of 30
    assert val_size == 3     # 10% of 30
    assert test_size == 3    # 10% of 30

def test_dataloader_creation(pipeline):
    """Test dataloader creation."""
    assert isinstance(pipeline.train_loader, torch.utils.data.DataLoader)
    assert isinstance(pipeline.val_loader, torch.utils.data.DataLoader)
    assert isinstance(pipeline.test_loader, torch.utils.data.DataLoader)
    
    # Check batch sizes
    train_batch = next(iter(pipeline.train_loader))
    assert len(train_batch['smiles']) == 4  # batch_size

def test_data_preprocessing(pipeline):
    """Test data preprocessing."""
    pipeline.preprocess_data()
    
    # Check that features are added to dataset
    sample = pipeline.dataset[0]
    assert 'fingerprint' in sample
    assert 'descriptors' in sample
    assert isinstance(sample['fingerprint'], torch.Tensor)
    assert isinstance(sample['descriptors'], torch.Tensor)

def test_get_batch(pipeline):
    """Test batch retrieval."""
    # Test train batch
    train_batch = pipeline.get_batch('train')
    assert isinstance(train_batch, dict)
    assert 'fingerprints' in train_batch
    assert 'descriptors' in train_batch
    assert train_batch['fingerprints'].shape[0] == 4  # batch_size
    
    # Test validation batch
    val_batch = pipeline.get_batch('val')
    assert isinstance(val_batch, dict)
    assert val_batch['fingerprints'].shape[0] == 4
    
    # Test invalid split
    with pytest.raises(ValueError):
        pipeline.get_batch('invalid')

def test_dataset_info(pipeline):
    """Test dataset information retrieval."""
    info = pipeline.get_dataset_info()
    assert isinstance(info, dict)
    assert info['type'] == 'zinc'
    assert info['total_size'] == 30
    assert info['train_size'] == 24
    assert info['val_size'] == 3
    assert info['test_size'] == 3
    assert info['batch_size'] == 4
    assert 'feature_dims' in info

def test_preprocessor_save_load(pipeline, tmp_path):
    """Test preprocessor state saving and loading."""
    # Save preprocessor state
    save_path = tmp_path / 'preprocessor.pt'
    pipeline.save_preprocessor(save_path)
    assert save_path.exists()
    
    # Create new pipeline
    new_pipeline = DataPipeline(
        dataset_type='zinc',
        data_path=pipeline.data_path,
        batch_size=4
    )
    
    # Load preprocessor state
    new_pipeline.load_preprocessor(save_path)
    
    # Check that states match
    assert new_pipeline.preprocessor.fingerprint_size == pipeline.preprocessor.fingerprint_size
    assert new_pipeline.preprocessor.radius == pipeline.preprocessor.radius
    assert new_pipeline.preprocessor.use_chirality == pipeline.preprocessor.use_chirality
    assert new_pipeline.preprocessor.normalize == pipeline.preprocessor.normalize

def test_property_stats(pipeline):
    """Test property statistics retrieval."""
    # Test property ranges
    ranges = pipeline.get_property_ranges()
    assert isinstance(ranges, dict)
    assert 'logp' in ranges
    assert 'tpsa' in ranges
    assert all(key in ranges['logp'] for key in ['min', 'max'])
    
    # Test property statistics
    stats = pipeline.get_property_stats()
    assert isinstance(stats, dict)
    assert 'logp' in stats
    assert 'tpsa' in stats
    assert all(key in stats['logp'] for key in ['mean', 'std', 'min', 'max'])

def test_pipeline_edge_cases():
    """Test pipeline edge cases."""
    # Test with invalid dataset type
    with pytest.raises(ValueError):
        DataPipeline('invalid_type', 'path/to/data')
    
    # Test with invalid split ratios
    with pytest.raises(ValueError):
        DataPipeline(
            'zinc',
            'path/to/data',
            train_split=0.5,
            val_split=0.3,
            test_split=0.3  # Sum > 1
        )
    
    # Test with invalid batch size
    with pytest.raises(ValueError):
        DataPipeline('zinc', 'path/to/data', batch_size=0)
    
    # Test with invalid number of workers
    with pytest.raises(ValueError):
        DataPipeline('zinc', 'path/to/data', num_workers=-1)

def test_pipeline_with_different_datasets(sample_data_path):
    """Test pipeline with different dataset types."""
    # Test with ChEMBL dataset
    chembl_pipeline = DataPipeline(
        dataset_type='chembl',
        data_path=sample_data_path,
        target_type='IC50'
    )
    assert isinstance(chembl_pipeline.dataset, ChEMBLDataset)
    
    # Test with MoleculeNet dataset
    moleculenet_pipeline = DataPipeline(
        dataset_type='moleculenet',
        data_path=sample_data_path,
        task='ESOL'
    )
    assert isinstance(moleculenet_pipeline.dataset, MoleculeNetDataset)
    
    # Test with custom dataset
    custom_pipeline = DataPipeline(
        dataset_type='custom',
        data_path=sample_data_path,
        smiles_column='smiles',
        property_columns=['logp', 'tpsa']
    )
    assert isinstance(custom_pipeline.dataset, CustomDataset) 