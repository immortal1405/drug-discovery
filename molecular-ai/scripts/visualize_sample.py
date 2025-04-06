import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # molecular-ai directory
raw_data_path = os.path.join(base_dir, 'data', 'raw', 'zinc_250k.csv')
output_image_path = os.path.join(base_dir, 'sample_molecule_zinc_0.png')

logger.info(f"Reading dataset from: {raw_data_path}")

try:
    # Read the CSV
    df = pd.read_csv(raw_data_path)

    if df.empty or 'SMILES' not in df.columns:
        logger.error("Dataset is empty or 'SMILES' column not found.")
    else:
        # Get the first SMILES string
        smiles = df['SMILES'].iloc[0]
        logger.info(f"Processing SMILES: {smiles}")

        # Create molecule object
        mol = Chem.MolFromSmiles(smiles)

        if mol:
            # Generate image data
            img = Draw.MolToImage(mol, size=(300, 300))
            # Save the image
            img.save(output_image_path)
            logger.info(f"Molecule image saved to: {output_image_path}")
        else:
            logger.error("Could not generate molecule from SMILES string.")

except FileNotFoundError:
    logger.error(f"Raw dataset file not found at: {raw_data_path}")
except Exception as e:
    logger.error(f"An error occurred: {str(e)}") 