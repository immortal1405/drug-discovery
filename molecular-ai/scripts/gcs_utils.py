#!/usr/bin/env python
# Utility functions for working with Google Cloud Storage

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

def is_gcs_path(path: str) -> bool:
    """Check if a path is a Google Cloud Storage path."""
    return path.startswith("gs://")

def download_from_gcs(gcs_path: str, local_path: Optional[str] = None) -> str:
    """
    Download a file from Google Cloud Storage to a local path.
    
    Args:
        gcs_path: GCS path in the format "gs://bucket-name/path/to/file"
        local_path: Local path to save the file. If None, a temporary file will be created.
        
    Returns:
        Local path where the file was saved
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage package is not installed. Run: pip install google-cloud-storage")
        raise
    
    if not is_gcs_path(gcs_path):
        logger.warning(f"Not a GCS path: {gcs_path}. Returning as is.")
        return gcs_path
    
    # Parse GCS path
    bucket_name = gcs_path.split("/")[2]
    blob_path = "/".join(gcs_path.split("/")[3:])
    
    # Set local path if not provided
    if local_path is None:
        # Create a temporary file with the same extension as the original file
        _, ext = os.path.splitext(blob_path)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        local_path = temp_file.name
        temp_file.close()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    
    # Download the file
    logger.info(f"Downloading {gcs_path} to {local_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded {gcs_path} to {local_path}")
    
    return local_path

def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_path: Local path of the file to upload
        gcs_path: GCS path in the format "gs://bucket-name/path/to/file"
        
    Returns:
        GCS path where the file was uploaded
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage package is not installed. Run: pip install google-cloud-storage")
        raise
    
    if not is_gcs_path(gcs_path):
        logger.error(f"Not a valid GCS path: {gcs_path}")
        raise ValueError(f"Not a valid GCS path: {gcs_path}")
    
    # Parse GCS path
    bucket_name = gcs_path.split("/")[2]
    blob_path = "/".join(gcs_path.split("/")[3:])
    
    # Upload the file
    logger.info(f"Uploading {local_path} to {gcs_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to {gcs_path}")
    
    return gcs_path

def load_model_from_path(model_path: str, model_class, device, **kwargs):
    """
    Load a model from a local path or GCS path.
    
    Args:
        model_path: Local path or GCS path to the model
        model_class: PyTorch model class to instantiate
        device: PyTorch device to load the model to
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Loaded PyTorch model
    """
    import torch
    
    # Download from GCS if needed
    if is_gcs_path(model_path):
        model_path = download_from_gcs(model_path)
    
    # Load model
    model = model_class(**kwargs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example: Download a file from GCS
    # local_path = download_from_gcs("gs://moleculargeneration-models/vae/final_vae_model.pt")
    # print(f"Downloaded to {local_path}")
    
    # Example: Upload a file to GCS
    # gcs_path = upload_to_gcs("local_file.txt", "gs://moleculargeneration-models/test/local_file.txt")
    # print(f"Uploaded to {gcs_path}")
    
    print("This is a utility module for working with Google Cloud Storage. Import it in your scripts.") 