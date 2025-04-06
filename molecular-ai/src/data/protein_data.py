"""
Protein data handling module for molecular generation models.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import PDB
from Bio.PDB import *
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
from deepchem.utils.docking_utils import prepare_inputs
from deepchem.utils.docking_utils import run_vina
from deepchem.utils.docking_utils import prepare_ligand
from deepchem.utils.docking_utils import prepare_receptor

logger = logging.getLogger(__name__)

class ProteinData:
    """Class for handling protein data and docking simulations."""
    
    def __init__(
        self,
        pdb_path: Union[str, Path],
        binding_pocket_path: Optional[Union[str, Path]] = None,
        constraints_path: Optional[Union[str, Path]] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None
    ):
        """
        Initialize protein data handler.
        
        Args:
            pdb_path: Path to PDB file
            binding_pocket_path: Path to binding pocket information
            constraints_path: Path to molecular constraints
            center: Center coordinates for docking box
            size: Size of docking box
        """
        self.pdb_path = Path(pdb_path)
        self.binding_pocket_path = Path(binding_pocket_path) if binding_pocket_path else None
        self.constraints_path = Path(constraints_path) if constraints_path else None
        self.center = center
        self.size = size
        
        # Load protein structure
        self.structure = self._load_pdb()
        self.model = self.structure[0]
        
        # Initialize DeepChem and AutoDock Vina
        self._setup_docking()
    
    def _load_pdb(self) -> PDB.Structure:
        """
        Load PDB structure.
        
        Returns:
            Bio.PDB.Structure object
        """
        parser = PDB.PDBParser()
        return parser.get_structure('protein', str(self.pdb_path))
    
    def _setup_docking(self) -> None:
        """Setup DeepChem and AutoDock Vina for docking."""
        # Prepare receptor
        self.receptor = prepare_receptor(str(self.pdb_path))
        
        # Get binding pocket if not provided
        if not self.center or not self.size:
            self.center, self.size = self._get_binding_pocket()
    
    def _get_binding_pocket(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get binding pocket information.
        
        Returns:
            Tuple of (center, size) coordinates
        """
        if self.binding_pocket_path:
            # Load from file
            pocket_data = pd.read_csv(self.binding_pocket_path)
            center = tuple(pocket_data['center'].values)
            size = tuple(pocket_data['size'].values)
        else:
            # Compute from structure
            coords = []
            for atom in self.model.get_atoms():
                coords.append(atom.get_coord())
            coords = np.array(coords)
            center = tuple(np.mean(coords, axis=0))
            size = tuple(np.max(coords, axis=0) - np.min(coords, axis=0))
        
        return center, size
    
    def get_binding_site_residues(self) -> List[Residue]:
        """
        Get residues in the binding site.
        
        Returns:
            List of binding site residues
        """
        residues = []
        for residue in self.model:
            if self._is_in_binding_site(residue):
                residues.append(residue)
        return residues
    
    def _is_in_binding_site(self, residue: Residue) -> bool:
        """
        Check if a residue is in the binding site.
        
        Args:
            residue: Bio.PDB.Residue object
            
        Returns:
            Whether residue is in binding site
        """
        for atom in residue:
            coord = atom.get_coord()
            if self._is_in_box(coord):
                return True
        return False
    
    def _is_in_box(self, coord: np.ndarray) -> bool:
        """
        Check if coordinates are in the docking box.
        
        Args:
            coord: 3D coordinates
            
        Returns:
            Whether coordinates are in box
        """
        for i in range(3):
            if abs(coord[i] - self.center[i]) > self.size[i] / 2:
                return False
        return True
    
    def get_molecular_constraints(self) -> Dict[str, Any]:
        """
        Get molecular constraints.
        
        Returns:
            Dictionary of constraints
        """
        if not self.constraints_path:
            return {}
        
        constraints = pd.read_csv(self.constraints_path)
        return constraints.to_dict('records')[0]
    
    def prepare_ligand_for_docking(
        self,
        smiles: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Prepare ligand for docking.
        
        Args:
            smiles: SMILES string
            output_path: Path to save prepared ligand
            
        Returns:
            Path to prepared ligand
        """
        if not output_path:
            output_path = self.pdb_path.parent / 'prepared_ligand.pdbqt'
        
        # Convert SMILES to 3D structure
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Prepare ligand for docking
        prepared_ligand = prepare_ligand(mol, str(output_path))
        return Path(prepared_ligand)
    
    def dock_ligand(
        self,
        ligand_path: Union[str, Path],
        exhaustiveness: int = 8,
        num_poses: int = 9
    ) -> List[Dict[str, Any]]:
        """
        Dock a ligand to the protein.
        
        Args:
            ligand_path: Path to ligand file
            exhaustiveness: Exhaustiveness of search
            num_poses: Number of poses to generate
            
        Returns:
            List of docking results
        """
        # Run AutoDock Vina
        results = run_vina(
            receptor=self.receptor,
            ligand=str(ligand_path),
            center=self.center,
            size=self.size,
            exhaustiveness=exhaustiveness,
            num_poses=num_poses
        )
        
        return results
    
    def predict_binding_affinity(
        self,
        smiles: str,
        model_type: str = 'deepchem'
    ) -> float:
        """
        Predict binding affinity using DeepChem models.
        
        Args:
            smiles: SMILES string
            model_type: Type of model to use
            
        Returns:
            Predicted binding affinity
        """
        if model_type == 'deepchem':
            # Use DeepChem's built-in models
            model = dc.models.GraphConvModel(
                n_tasks=1,
                mode='regression',
                batch_size=32
            )
            
            # Prepare input
            featurizer = dc.feat.ConvMolFeaturizer()
            mol = Chem.MolFromSmiles(smiles)
            features = featurizer.featurize([mol])
            
            # Make prediction
            prediction = model.predict(features)
            return float(prediction[0][0])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_protein_info(self) -> Dict[str, Any]:
        """
        Get information about the protein.
        
        Returns:
            Dictionary of protein information
        """
        return {
            'pdb_id': self.pdb_path.stem,
            'num_chains': len(list(self.model)),
            'num_residues': len(list(self.model.get_residues())),
            'num_atoms': len(list(self.model.get_atoms())),
            'binding_site_residues': len(self.get_binding_site_residues()),
            'binding_site_center': self.center,
            'binding_site_size': self.size
        }
    
    def save_binding_site(self, output_path: Union[str, Path]) -> None:
        """
        Save binding site information.
        
        Args:
            output_path: Path to save binding site
        """
        output_path = Path(output_path)
        binding_site = self.get_binding_site_residues()
        
        # Create new structure with only binding site
        io = PDBIO()
        structure = PDB.Structure.Structure('binding_site')
        model = PDB.Model.Model(0)
        structure.add(model)
        
        for residue in binding_site:
            model.add(residue)
        
        io.set_structure(structure)
        io.save(str(output_path)) 