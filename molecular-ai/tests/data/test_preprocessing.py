"""
Tests for molecular data preprocessing.
"""

import pytest
import numpy as np
import torch
from rdkit import Chem
from src.data.preprocessing import MolecularPreprocessor

@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing."""
    smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

@pytest.fixture
def preprocessor():
    """Create preprocessor instance for testing."""
    return MolecularPreprocessor(
        fingerprint_size=1024,
        radius=2,
        use_chirality=True,
        normalize=True
    )

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = MolecularPreprocessor()
    assert preprocessor.fingerprint_size == 2048
    assert preprocessor.radius == 2
    assert preprocessor.use_chirality is True
    assert preprocessor.normalize is True
    assert isinstance(preprocessor.feature_stats, dict)

def test_compute_fingerprint(preprocessor, sample_molecules):
    """Test fingerprint computation."""
    mol = sample_molecules[0]
    fingerprint = preprocessor.compute_fingerprint(mol)
    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape == (1, 1024)
    assert fingerprint.dtype == np.float64

def test_compute_descriptors(preprocessor, sample_molecules):
    """Test descriptor computation."""
    mol = sample_molecules[0]
    descriptors = preprocessor.compute_descriptors(mol)
    assert isinstance(descriptors, dict)
    assert all(key in descriptors for key in [
        'molecular_weight', 'logp', 'tpsa', 'qed',
        'rotatable_bonds', 'hbd', 'hba', 'rings',
        'aromatic_rings'
    ])
    assert all(isinstance(value, float) for value in descriptors.values())

def test_compute_3d_coordinates(preprocessor, sample_molecules):
    """Test 3D coordinate computation."""
    mol = sample_molecules[0]
    coordinates = preprocessor.compute_3d_coordinates(mol)
    assert isinstance(coordinates, np.ndarray)
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 3  # x, y, z coordinates

def test_preprocess_molecule(preprocessor, sample_molecules):
    """Test single molecule preprocessing."""
    mol = sample_molecules[0]
    features = preprocessor.preprocess_molecule(mol, compute_3d=True)
    assert isinstance(features, dict)
    assert 'fingerprint' in features
    assert 'descriptors' in features
    assert 'coordinates' in features
    assert isinstance(features['fingerprint'], np.ndarray)
    assert isinstance(features['descriptors'], dict)
    assert isinstance(features['coordinates'], np.ndarray)

def test_preprocess_batch(preprocessor, sample_molecules):
    """Test batch preprocessing."""
    features = preprocessor.preprocess_batch(sample_molecules, compute_3d=True)
    assert isinstance(features, dict)
    assert 'fingerprints' in features
    assert 'descriptors' in features
    assert 'coordinates' in features
    assert features['fingerprints'].shape == (3, 1024)
    assert features['descriptors'].shape == (3, 9)  # 9 descriptors
    assert features['coordinates'].shape[0] == 3  # 3 molecules

def test_fit_transform(preprocessor, sample_molecules):
    """Test fit and transform functionality."""
    tensors = preprocessor.fit_transform(sample_molecules, compute_3d=True)
    assert isinstance(tensors, dict)
    assert 'fingerprints' in tensors
    assert 'descriptors' in tensors
    assert 'coordinates' in tensors
    assert isinstance(tensors['fingerprints'], torch.Tensor)
    assert isinstance(tensors['descriptors'], torch.Tensor)
    assert isinstance(tensors['coordinates'], torch.Tensor)
    
    # Check normalization
    descriptors = tensors['descriptors'].numpy()
    assert np.allclose(np.mean(descriptors, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(descriptors, axis=0), 1, atol=1e-10)

def test_inverse_transform(preprocessor, sample_molecules):
    """Test inverse transformation."""
    # First fit and transform
    tensors = preprocessor.fit_transform(sample_molecules)
    original_descriptors = preprocessor.inverse_transform(tensors['descriptors'])
    
    # Check shape and type
    assert isinstance(original_descriptors, np.ndarray)
    assert original_descriptors.shape == (3, 9)  # 3 molecules, 9 descriptors
    
    # Check that inverse transform restores original scale
    descriptors = tensors['descriptors'].numpy()
    restored = preprocessor.inverse_transform(torch.FloatTensor(descriptors))
    assert np.allclose(restored, original_descriptors)

def test_preprocessor_edge_cases():
    """Test preprocessor edge cases."""
    preprocessor = MolecularPreprocessor(normalize=False)
    
    # Test with empty molecule list
    with pytest.raises(ValueError):
        preprocessor.fit([])
    
    # Test with invalid molecule
    invalid_mol = Chem.Mol()
    with pytest.raises(Exception):
        preprocessor.compute_fingerprint(invalid_mol)
    
    # Test with None molecule
    with pytest.raises(Exception):
        preprocessor.compute_descriptors(None)

def test_preprocessor_parameters():
    """Test preprocessor parameter variations."""
    # Test different fingerprint sizes
    preprocessor = MolecularPreprocessor(fingerprint_size=512)
    mol = Chem.MolFromSmiles('CCO')
    fingerprint = preprocessor.compute_fingerprint(mol)
    assert fingerprint.shape == (1, 512)
    
    # Test different radius
    preprocessor = MolecularPreprocessor(radius=3)
    fingerprint = preprocessor.compute_fingerprint(mol)
    assert fingerprint.shape == (1, 2048)
    
    # Test without chirality
    preprocessor = MolecularPreprocessor(use_chirality=False)
    fingerprint = preprocessor.compute_fingerprint(mol)
    assert fingerprint.shape == (1, 2048)
    
    # Test without normalization
    preprocessor = MolecularPreprocessor(normalize=False)
    mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC(=O)O')]
    tensors = preprocessor.fit_transform(mols)
    descriptors = tensors['descriptors'].numpy()
    assert not np.allclose(np.mean(descriptors, axis=0), 0, atol=1e-10) 