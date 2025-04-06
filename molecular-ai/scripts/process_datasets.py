import os
import logging
from pathlib import Path
import sys

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset_manager import DatasetManager

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_processing.log')
        ]
    )

def main():
    """Main function to process all datasets."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize dataset manager
        data_dir = project_root / "data"
        manager = DatasetManager(str(data_dir))
        
        # Process all datasets
        logger.info("Starting dataset processing...")
        manager.process_all_datasets()
        logger.info("Dataset processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing datasets: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 