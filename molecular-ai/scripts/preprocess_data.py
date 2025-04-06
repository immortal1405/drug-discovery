import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Tuple, List
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from deepchem.feat import ConvMolFeaturizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_supported_molecule(mol) -> bool:
    """Check if all atoms in molecule are supported by our processing pipeline.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        bool: True if molecule contains only supported atoms
    """
    # List of supported elements (add or remove as needed)
    supported_elements = {
        1, 6, 7, 8, 9, 15, 16, 17, 35, 53  # H, C, N, O, F, P, S, Cl, Br, I
    }
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in supported_elements:
            return False
        if abs(atom.GetFormalCharge()) > 2:  # Skip molecules with high charges
            return False
    return True

class MolecularDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        """Initialize the molecular dataset."""
        # Remove featurizer init
        # self.featurizer = ConvMolFeaturizer()
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        """List of raw data files."""
        return ['zinc_250k.csv', 'chembl.csv', 'qm9.csv', 'tox21.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        """List of processed data files."""
        return ['train.pt', 'val.pt', 'test.pt']
    
    def process(self):
        """Process the raw data into PyTorch Geometric format."""
        logger.info("Processing datasets...")
        
        all_data = []
        for raw_file in self.raw_file_names:
            df = pd.read_csv(os.path.join(self.raw_dir, raw_file))
            logger.info(f"Processing {raw_file} with {len(df)} molecules")
            
            for _, row in df.iterrows():
                try:
                    # Convert SMILES to RDKit molecule
                    mol = Chem.MolFromSmiles(row['SMILES'])
                    if mol is None:
                        logger.warning(f"Failed to parse SMILES: {row['SMILES']}")
                        continue
                    
                    # Check if molecule contains only supported atoms
                    if not is_supported_molecule(mol):
                        logger.warning(f"Skipping molecule with unsupported atoms: {row['SMILES']}")
                        continue
                    
                    # Add hydrogens and sanitize
                    try:
                        mol = Chem.AddHs(mol)
                        Chem.SanitizeMol(mol)
                    except Exception as e:
                        logger.warning(f"Failed to add hydrogens or sanitize: {str(e)}")
                        continue
                    
                    # Generate 3D coordinates with multiple attempts
                    coords_generated = False
                    for seed in [42, 123, 456, 789]:  # Try multiple random seeds
                        try:
                            # Try ETKDG first (better quality)
                            params = AllChem.ETKDGv3()
                            params.randomSeed = seed
                            AllChem.EmbedMolecule(mol, params=params)
                            
                            if mol.GetNumConformers() > 0:
                                coords_generated = True
                                break
                        except Exception as e:
                            try:
                                # Fallback to simpler distance geometry
                                AllChem.EmbedMolecule(mol, randomSeed=seed)
                                if mol.GetNumConformers() > 0:
                                    coords_generated = True
                                    break
                            except Exception as e:
                                continue
                    
                    if not coords_generated:
                        logger.warning(f"Failed to generate 3D coordinates for molecule")
                        continue
                    
                    try:
                        # Try MMFF94 first (more accurate)
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    except:
                        try:
                            # Fallback to UFF
                            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                        except Exception as e:
                            logger.warning(f"Failed to optimize geometry: {str(e)}")
                            # Continue anyway since we have initial 3D coordinates
                    
                    # --- Revert to Manual Node Feature Calculation --- 
                    node_features = []
                    for atom in mol.GetAtoms():
                        # One-hot encode atom type
                        atom_type = [0] * 100  # Support up to atomic number 100
                        try:
                            atom_type[atom.GetAtomicNum() - 1] = 1
                        except IndexError:
                            pass # Keep all zeros if atomic number > 100
                        
                        # One-hot encode degree
                        degree = [0] * 6  # Support up to degree 5
                        degree[min(atom.GetDegree(), 5)] = 1
                        
                        # One-hot encode formal charge
                        formal_charge = [0] * 5  # -2, -1, 0, +1, +2
                        charge_idx = min(max(atom.GetFormalCharge() + 2, 0), 4)
                        formal_charge[charge_idx] = 1
                        
                        # One-hot encode hybridization
                        hybridization = [0] * 5  # sp, sp2, sp3, sp3d, sp3d2
                        try:
                            hyb_type = str(atom.GetHybridization())
                            if hyb_type == 'SP': hybridization[0] = 1
                            elif hyb_type == 'SP2': hybridization[1] = 1
                            elif hyb_type == 'SP3': hybridization[2] = 1
                            elif hyb_type == 'SP3D': hybridization[3] = 1
                            elif hyb_type == 'SP3D2': hybridization[4] = 1
                        except ValueError: # Handle potential errors from GetHybridization
                            pass  # Keep all zeros if hybridization is unknown or error occurs
                        
                        # Additional features
                        additional = [
                            float(atom.GetIsAromatic()),
                            float(atom.GetNumRadicalElectrons()),
                            float(atom.GetTotalNumHs()),
                            float(atom.IsInRing())
                        ]
                        
                        # Combine all features
                        features = (
                            atom_type +
                            degree +
                            formal_charge +
                            hybridization +
                            additional
                        )
                        node_features.append(features)
                    # --- End of Manual Node Features --- 
                    
                    # --- Manual Edge Feature Calculation --- 
                    edge_index = []
                    edge_features = []
                    for bond in mol.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        edge_index.extend([[i, j], [j, i]])  # Add both directions
                        
                        # One-hot encode bond type
                        bond_type = [0] * 4  # single, double, triple, aromatic
                        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE: bond_type[0] = 1
                        elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE: bond_type[1] = 1
                        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE: bond_type[2] = 1
                        elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC: bond_type[3] = 1
                        
                        # Additional bond features
                        additional = [
                            float(bond.GetIsConjugated()),
                            float(bond.IsInRing())
                        ]
                        
                        # Combine all features
                        edge_features_single = bond_type + additional
                        edge_features.extend([edge_features_single, edge_features_single])  # Add for both directions
                    # --- End of manual edge features --- 
                    
                    # Convert to tensors
                    node_features = torch.tensor(node_features, dtype=torch.float)
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
                    edge_features = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 6), dtype=torch.float)
                    
                    # Create PyTorch Geometric Data object (with graph-level y)
                    # Reshape y to [1, num_tasks] to indicate graph-level target
                    target_y = torch.tensor([
                        row['MolecularWeight'],
                        row['LogP'],
                        row['RotatableBonds'],
                        row['HBondAcceptors'],
                        row['HBondDonors'],
                        row['FormalCharge']
                    ], dtype=torch.float).unsqueeze(0) # Add batch dimension

                    data = Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=target_y,
                        smiles=row['SMILES']
                    )
                    
                    all_data.append(data)
                    
                except Exception as e:
                    logger.warning(f"Error processing molecule: {str(e)}")
                    continue
        
        # Split dataset
        train_val_data, test_data = train_test_split(
            all_data, test_size=0.2, random_state=42
        )
        train_data, val_data = train_test_split(
            train_val_data, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
        )
        
        # Save processed datasets
        torch.save(train_data, os.path.join(self.processed_dir, 'train.pt'))
        torch.save(val_data, os.path.join(self.processed_dir, 'val.pt'))
        torch.save(test_data, os.path.join(self.processed_dir, 'test.pt'))
        
        logger.info(f"Processed {len(all_data)} molecules")
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

def main():
    """Main function to preprocess the datasets."""
    # Create dataset
    dataset = MolecularDataset(root='data')
    
    # Process the data
    dataset.process()

if __name__ == "__main__":
    main() 