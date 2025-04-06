"""
Tests for protein data handling module.
"""

import pytest
import numpy as np
from pathlib import Path
from Bio import PDB
from src.data.protein_data import ProteinData
import pandas as pd
from unittest.mock import patch

@pytest.fixture
def sample_pdb_path(tmp_path) -> Path:
    """Create a sample PDB file for testing."""
    # Create a simple PDB file with a binding site
    pdb_content = """
ATOM      1  N   ALA A   1      27.470  11.280  10.410  1.00 20.00           N
ATOM      2  CA  ALA A   1      26.960  10.100  11.200  1.00 20.00           C
ATOM      3  C   ALA A   1      25.470  10.100  11.200  1.00 20.00           C
ATOM      4  O   ALA A   1      24.960   9.000  11.200  1.00 20.00           O
ATOM      5  CB  ALA A   1      27.470   8.800  10.410  1.00 20.00           C
ATOM      6  N   ALA A   2      24.960  11.280  11.200  1.00 20.00           N
ATOM      7  CA  ALA A   2      23.470  11.280  11.200  1.00 20.00           C
ATOM      8  C   ALA A   2      22.960  10.100  10.410  1.00 20.00           C
ATOM      9  O   ALA A   2      21.960  10.100  10.410  1.00 20.00           O
ATOM     10  CB  ALA A   2      23.470  12.560  12.410  1.00 20.00           C
"""
    pdb_path = tmp_path / 'sample.pdb'
    pdb_path.write_text(pdb_content)
    return pdb_path

@pytest.fixture
def sample_binding_pocket_path(tmp_path) -> Path:
    """Create sample binding pocket information."""
    pocket_data = {
        'center': [25.0, 10.0, 11.0],
        'size': [10.0, 5.0, 5.0]
    }
    pocket_path = tmp_path / 'binding_pocket.csv'
    pd.DataFrame(pocket_data).to_csv(pocket_path, index=False)
    return pocket_path

@pytest.fixture
def sample_constraints_path(tmp_path) -> Path:
    """Create sample molecular constraints."""
    constraints = {
        'max_molecular_weight': 500.0,
        'min_logp': -2.0,
        'max_logp': 5.0,
        'max_hbd': 5,
        'max_hba': 10,
        'max_rotatable_bonds': 8
    }
    constraints_path = tmp_path / 'constraints.csv'
    pd.DataFrame(constraints).to_csv(constraints_path, index=False)
    return constraints_path

@pytest.fixture
def protein_data(sample_pdb_path, sample_binding_pocket_path, sample_constraints_path):
    """Create ProteinData instance for testing."""
    return ProteinData(
        pdb_path=sample_pdb_path,
        binding_pocket_path=sample_binding_pocket_path,
        constraints_path=sample_constraints_path
    )

def test_protein_data_initialization(protein_data):
    """Test protein data initialization."""
    assert isinstance(protein_data.structure, PDB.Structure)
    assert isinstance(protein_data.model, PDB.Model)
    assert protein_data.center is not None
    assert protein_data.size is not None

def test_binding_pocket_loading(protein_data):
    """Test binding pocket loading."""
    center, size = protein_data._get_binding_pocket()
    assert len(center) == 3
    assert len(size) == 3
    assert all(isinstance(x, float) for x in center)
    assert all(isinstance(x, float) for x in size)

def test_binding_site_residues(protein_data):
    """Test binding site residue identification."""
    residues = protein_data.get_binding_site_residues()
    assert isinstance(residues, list)
    assert all(isinstance(r, PDB.Residue) for r in residues)

def test_molecular_constraints(protein_data):
    """Test molecular constraints loading."""
    constraints = protein_data.get_molecular_constraints()
    assert isinstance(constraints, dict)
    assert 'max_molecular_weight' in constraints
    assert 'min_logp' in constraints
    assert 'max_logp' in constraints

def test_ligand_preparation(protein_data, tmp_path):
    """Test ligand preparation for docking."""
    smiles = 'CCO'  # ethanol
    output_path = tmp_path / 'prepared_ligand.pdbqt'
    
    prepared_path = protein_data.prepare_ligand_for_docking(
        smiles,
        output_path=output_path
    )
    assert prepared_path.exists()
    assert prepared_path.suffix == '.pdbqt'

@patch('deepchem.utils.docking_utils.run_vina')
def test_ligand_docking(mock_run_vina, protein_data, tmp_path):
    """Test ligand docking."""
    # Mock docking results
    mock_results = [
        {'score': -8.5, 'pose': 'pose1.pdbqt'},
        {'score': -7.2, 'pose': 'pose2.pdbqt'}
    ]
    mock_run_vina.return_value = mock_results
    
    # Test docking
    ligand_path = tmp_path / 'test_ligand.pdbqt'
    ligand_path.touch()
    
    results = protein_data.dock_ligand(
        ligand_path,
        exhaustiveness=8,
        num_poses=2
    )
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert all('score' in r for r in results)
    assert all('pose' in r for r in results)

@patch('deepchem.models.GraphConvModel')
def test_binding_affinity_prediction(mock_model, protein_data):
    """Test binding affinity prediction."""
    # Mock model prediction
    mock_model.return_value.predict.return_value = np.array([[0.5]])
    
    smiles = 'CCO'  # ethanol
    affinity = protein_data.predict_binding_affinity(smiles)
    
    assert isinstance(affinity, float)
    assert affinity == 0.5

def test_protein_info(protein_data):
    """Test protein information retrieval."""
    info = protein_data.get_protein_info()
    
    assert isinstance(info, dict)
    assert 'pdb_id' in info
    assert 'num_chains' in info
    assert 'num_residues' in info
    assert 'num_atoms' in info
    assert 'binding_site_residues' in info
    assert 'binding_site_center' in info
    assert 'binding_site_size' in info

def test_binding_site_saving(protein_data, tmp_path):
    """Test binding site saving."""
    output_path = tmp_path / 'binding_site.pdb'
    protein_data.save_binding_site(output_path)
    
    assert output_path.exists()
    # Verify saved structure
    structure = PDB.PDBParser().get_structure('binding_site', str(output_path))
    assert len(list(structure.get_residues())) > 0

def test_protein_data_edge_cases():
    """Test protein data edge cases."""
    # Test with invalid PDB file
    with pytest.raises(Exception):
        ProteinData('invalid.pdb')
    
    # Test with invalid binding pocket file
    with pytest.raises(Exception):
        ProteinData('sample.pdb', binding_pocket_path='invalid.csv')
    
    # Test with invalid constraints file
    with pytest.raises(Exception):
        ProteinData('sample.pdb', constraints_path='invalid.csv')
    
    # Test with invalid SMILES
    protein_data = ProteinData('sample.pdb')
    with pytest.raises(Exception):
        protein_data.predict_binding_affinity('invalid_smiles') 