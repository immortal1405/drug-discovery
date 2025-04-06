#!/usr/bin/env python3
"""
Simple molecule generator script that outputs SMILES strings to a file.
This script is a fallback when the more complex generator has issues.
"""

import sys
import os
import random
import argparse
from pathlib import Path

# Drug-like SMILES templates
SMILES_TEMPLATES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
    "CN1C2=C(C=CC=C2)C(=O)N(C1=O)C", # Amobarbital
    "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1", # Salbutamol
    "NC(=O)C1=C(O)C=CC=C1O", # Paracetamol
    "COC1=CC=C(CCN2CCN(CC2)C3=C(C)N=CC=C3)C=C1", # Mirtazapine
    "C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3", # Warfarin
    "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN=CC2=CC=NC=C2", # Sulfamethoxazole
    "CC1=C(C=C(C=C1)S(=O)(=O)N)CC2=CC=C(C=C2)CC3C(=O)NC(=O)S3", # Gliclazide
    "ClC1=C(C=CC=C1)C2=NNC(=O)C3=CC=CC=C3N2", # Chlorzoxazone
    "CC1=CC=C(C=C1)C(=O)C2=C(O)C3=C(OC2=O)C=CC=C3", # Warfarin
    "CC1=C(C=C(N)C=C1)C(=O)O", # p-Aminosalicylic acid
    "COC1=CC2=C(C=C1OC)C(=O)C(CC2)CC3C(=O)NC(=O)S3", # Glibenclamide
    "CC(=O)C1=CC=C(C=C1)OC(=O)C", # Benzyl acetate
    "NC(=O)C1=C(O)C=CC=C1", # Salicylamide
    "CN1C=NC2=C1C(=O)NC(=O)N2C", # Caffeine
    "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2", # Propranolol
    "C1=CC=CC(=C1)C(=O)NC2=CC=C(C=C2)Cl", # Bupropion
    "CC1=C(C=CC=C1)C(=O)C2=C(O)C3=C(OC2=O)C=CC=C3", # Warfarin
]

def generate_molecules(num_samples, output_file):
    """Generate random drug-like molecules and save to file"""
    print(f"Generating {num_samples} molecules...")
    
    # Generate random variations of the templates
    generated_smiles = []
    for i in range(num_samples):
        # Select a random template
        base_smiles = random.choice(SMILES_TEMPLATES)
        
        # Add random modifications
        if random.random() < 0.7:  # 70% chance of modification
            # Insert a random atom at a random position
            atoms = ["C", "N", "O", "S", "F", "Cl"]
            atom = random.choice(atoms)
            pos = random.randint(0, len(base_smiles) - 1)
            modified_smiles = base_smiles[:pos] + atom + base_smiles[pos:]
        else:
            modified_smiles = base_smiles
        
        generated_smiles.append(modified_smiles)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for smiles in generated_smiles:
            f.write(f"{smiles}\n")
    
    print(f"Successfully generated {len(generated_smiles)} molecules")
    print(f"Saved to {output_file}")
    return generated_smiles

def main():
    parser = argparse.ArgumentParser(description='Simple Molecule Generator')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of molecules to generate')
    parser.add_argument('--output_file', type=str, default='generated_smiles.txt', help='Output file path')
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate molecules
    generated_smiles = generate_molecules(args.num_samples, args.output_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 