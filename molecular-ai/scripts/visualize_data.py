import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data visualizer.
        
        Args:
            data_dir: Directory containing the processed datasets
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8')  # Using a built-in style
        sns.set_palette("husl")
        
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a dataset from CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame containing the dataset
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def plot_molecular_weight_distribution(self, datasets: list[str]):
        """Plot molecular weight distribution for multiple datasets.
        
        Args:
            datasets: List of dataset filenames to compare
        """
        plt.figure(figsize=(12, 6))
        
        for dataset in datasets:
            df = self.load_dataset(dataset)
            sns.kdeplot(data=df['MolecularWeight'], label=dataset.replace('.csv', ''))
        
        plt.title('Molecular Weight Distribution Across Datasets')
        plt.xlabel('Molecular Weight (Da)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(self.output_dir / 'molecular_weight_distribution.png')
        plt.close()
        
    def plot_logp_distribution(self, datasets: list[str]):
        """Plot LogP distribution for multiple datasets.
        
        Args:
            datasets: List of dataset filenames to compare
        """
        plt.figure(figsize=(12, 6))
        
        for dataset in datasets:
            df = self.load_dataset(dataset)
            sns.kdeplot(data=df['LogP'], label=dataset.replace('.csv', ''))
        
        plt.title('LogP Distribution Across Datasets')
        plt.xlabel('LogP')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(self.output_dir / 'logp_distribution.png')
        plt.close()
        
    def plot_rotatable_bonds_distribution(self, datasets: list[str]):
        """Plot distribution of rotatable bonds for multiple datasets.
        
        Args:
            datasets: List of dataset filenames to compare
        """
        plt.figure(figsize=(12, 6))
        
        for dataset in datasets:
            df = self.load_dataset(dataset)
            sns.kdeplot(data=df['RotatableBonds'], label=dataset.replace('.csv', ''))
        
        plt.title('Rotatable Bonds Distribution Across Datasets')
        plt.xlabel('Number of Rotatable Bonds')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(self.output_dir / 'rotatable_bonds_distribution.png')
        plt.close()
        
    def plot_hbond_distribution(self, datasets: list[str]):
        """Plot hydrogen bond donor/acceptor distribution for multiple datasets.
        
        Args:
            datasets: List of dataset filenames to compare
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for dataset in datasets:
            df = self.load_dataset(dataset)
            sns.kdeplot(data=df['HBondAcceptors'], label=dataset.replace('.csv', ''), ax=ax1)
            sns.kdeplot(data=df['HBondDonors'], label=dataset.replace('.csv', ''), ax=ax2)
        
        ax1.set_title('Hydrogen Bond Acceptors Distribution')
        ax1.set_xlabel('Number of HBond Acceptors')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        ax2.set_title('Hydrogen Bond Donors Distribution')
        ax2.set_xlabel('Number of HBond Donors')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hbond_distribution.png')
        plt.close()
        
    def plot_tox21_correlations(self):
        """Plot correlation matrix for Tox21 dataset properties."""
        df = self.load_dataset('tox21.csv')
        
        # Select toxicity-related columns
        tox_cols = [col for col in df.columns if col.startswith(('NR-', 'SR-'))]
        
        # Create correlation matrix
        corr_matrix = df[tox_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Tox21 Dataset Property Correlations')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tox21_correlations.png')
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all visualizations for the datasets."""
        datasets = ['zinc_250k.csv', 'chembl.csv', 'qm9.csv', 'tox21.csv']
        
        try:
            logger.info("Generating molecular weight distribution plot...")
            self.plot_molecular_weight_distribution(datasets)
            
            logger.info("Generating LogP distribution plot...")
            self.plot_logp_distribution(datasets)
            
            logger.info("Generating rotatable bonds distribution plot...")
            self.plot_rotatable_bonds_distribution(datasets)
            
            logger.info("Generating hydrogen bond distribution plot...")
            self.plot_hbond_distribution(datasets)
            
            logger.info("Generating Tox21 correlations plot...")
            self.plot_tox21_correlations()
            
            logger.info("All visualizations generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to generate all visualizations."""
    visualizer = DataVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 