import torch
import logging

logger = logging.getLogger(__name__)

def load_processed_data(file_path: str):
    """Loads processed PyTorch Geometric dataset from a .pt file."""
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: Processed data file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise 