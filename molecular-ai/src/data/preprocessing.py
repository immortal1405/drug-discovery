"""
Data preprocessing module for molecular data transformations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

logger = logging.getLogger(__name__)

class MolecularPreprocessor:
    """Preprocessor for molecular data."""
    
    def __init__(
        self,
        fingerprint_size: int = 2048,
        radius: int = 2,
        use_chirality: bool = True,
        normalize: bool = True
    ):
        """
        Initialize molecular preprocessor.
        
        Args:
            fingerprint_size: Size of Morgan fingerprint
            radius: Radius for Morgan fingerprint
            use_chirality: Whether to use chirality in fingerprint
            normalize: Whether to normalize features
        """
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        self.use_chirality = use_chirality
        self.normalize = normalize
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def compute_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute Morgan fingerprint for a molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Morgan fingerprint as numpy array
        """
        fp = GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.fingerprint_size,
            useChirality=self.use_chirality
        )
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def compute_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute molecular descriptors.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary of molecular descriptors
        """
        return {
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'qed': QED.default(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rings': Descriptors.RingCount(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol)
        }
    
    def compute_3d_coordinates(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute 3D coordinates for a molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            3D coordinates as numpy array
        """
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return np.array(mol.GetConformer().GetPositions())
    
    def preprocess_molecule(
        self,
        mol: Chem.Mol,
        compute_3d: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess a single molecule.
        
        Args:
            mol: RDKit molecule
            compute_3d: Whether to compute 3D coordinates
            
        Returns:
            Dictionary of preprocessed features
        """
        features = {
            'fingerprint': self.compute_fingerprint(mol),
            'descriptors': self.compute_descriptors(mol)
        }
        
        if compute_3d:
            features['coordinates'] = self.compute_3d_coordinates(mol)
        
        return features
    
    def preprocess_batch(
        self,
        molecules: List[Chem.Mol],
        compute_3d: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of molecules.
        
        Args:
            molecules: List of RDKit molecules
            compute_3d: Whether to compute 3D coordinates
            
        Returns:
            Dictionary of preprocessed features
        """
        fingerprints = []
        descriptors = []
        coordinates = [] if compute_3d else None
        
        for mol in molecules:
            features = self.preprocess_molecule(mol, compute_3d)
            fingerprints.append(features['fingerprint'])
            descriptors.append(list(features['descriptors'].values()))
            if compute_3d:
                coordinates.append(features['coordinates'])
        
        batch_features = {
            'fingerprints': np.array(fingerprints),
            'descriptors': np.array(descriptors)
        }
        
        if compute_3d:
            batch_features['coordinates'] = np.array(coordinates)
        
        return batch_features
    
    def fit(self, molecules: List[Chem.Mol]) -> None:
        """
        Fit preprocessor on a set of molecules.
        
        Args:
            molecules: List of RDKit molecules
        """
        descriptors = []
        for mol in molecules:
            desc = self.compute_descriptors(mol)
            descriptors.append(list(desc.values()))
        
        descriptors = np.array(descriptors)
        self.feature_stats = {
            'mean': np.mean(descriptors, axis=0),
            'std': np.std(descriptors, axis=0)
        }
    
    def transform(
        self,
        molecules: List[Chem.Mol],
        compute_3d: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Transform molecules to tensors.
        
        Args:
            molecules: List of RDKit molecules
            compute_3d: Whether to compute 3D coordinates
            
        Returns:
            Dictionary of feature tensors
        """
        features = self.preprocess_batch(molecules, compute_3d)
        
        # Convert to tensors
        tensors = {
            'fingerprints': torch.FloatTensor(features['fingerprints']),
            'descriptors': torch.FloatTensor(features['descriptors'])
        }
        
        if compute_3d:
            tensors['coordinates'] = torch.FloatTensor(features['coordinates'])
        
        # Normalize if enabled
        if self.normalize and self.feature_stats:
            tensors['descriptors'] = (
                tensors['descriptors'] - self.feature_stats['mean']
            ) / self.feature_stats['std']
        
        return tensors
    
    def fit_transform(
        self,
        molecules: List[Chem.Mol],
        compute_3d: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fit preprocessor and transform molecules.
        
        Args:
            molecules: List of RDKit molecules
            compute_3d: Whether to compute 3D coordinates
            
        Returns:
            Dictionary of feature tensors
        """
        self.fit(molecules)
        return self.transform(molecules, compute_3d)
    
    def inverse_transform(
        self,
        descriptors: torch.Tensor
    ) -> np.ndarray:
        """
        Transform normalized descriptors back to original scale.
        
        Args:
            descriptors: Normalized descriptor tensor
            
        Returns:
            Descriptors in original scale
        """
        if not self.normalize or not self.feature_stats:
            return descriptors.numpy()
        
        return (
            descriptors.numpy() * self.feature_stats['std'] +
            self.feature_stats['mean']
        ) 