"""
Test cases for the model factory.
"""

import os
import pytest
import torch
import numpy as np
from src.models.model_factory import ModelFactory
from src.models.vae.vae_model import VAE
from src.models.gan.gan_model import GAN
from src.models.gnn.gnn_model import GNN

@pytest.fixture
def model_params():
    """Fixture for common model parameters."""
    return {
        'input_dim': 100,
        'hidden_dim': 256,
        'latent_dim': 64,
        'output_dim': 100,
        'dropout': 0.1
    }

@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def batch_data(model_params, device):
    """Fixture for test batch data."""
    batch_size = 32
    return {
        'x': torch.randn(batch_size, model_params['input_dim'], device=device),
        'edge_index': torch.randint(0, model_params['input_dim'], (2, 100), device=device)
    }

def test_model_registry():
    """Test model registry functionality."""
    # Check if all expected models are registered
    available_models = ModelFactory.get_available_models()
    assert 'vae' in available_models
    assert 'gan' in available_models
    assert 'gnn' in available_models
    
    # Test model info retrieval
    vae_info = ModelFactory.get_model_info('vae')
    assert vae_info['name'] == 'vae'
    assert vae_info['class'] == 'VAE'
    assert vae_info['description'] is not None
    assert vae_info['parameters'] is not None

def test_create_models(model_params, device):
    """Test model creation for all model types."""
    # Test VAE creation
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    assert isinstance(vae_model, VAE)
    assert vae_model.device == device
    
    # Test GAN creation
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    assert isinstance(gan_model, GAN)
    assert gan_model.device == device
    
    # Test GNN creation
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    assert isinstance(gnn_model, GNN)
    assert gnn_model.device == device

def test_invalid_model_type():
    """Test handling of invalid model type."""
    with pytest.raises(ValueError, match="Model type 'invalid_model' not registered"):
        ModelFactory.create_model(
            model_type='invalid_model',
            input_dim=100,
            hidden_dim=256,
            latent_dim=64,
            output_dim=100
        )

def test_model_save_load(model_params, device, tmp_path):
    """Test model saving and loading functionality."""
    # Create and save a VAE model
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    
    # Generate some dummy data
    batch_size = 32
    x = torch.randn(batch_size, model_params['input_dim'], device=device)
    
    # Save model
    save_path = os.path.join(tmp_path, 'vae_model.pt')
    vae_model.save(save_path)
    assert os.path.exists(save_path)
    
    # Load model
    loaded_model = ModelFactory.load_model(
        model_type='vae',
        model_path=save_path,
        device=device,
        **model_params
    )
    
    # Verify loaded model
    assert isinstance(loaded_model, VAE)
    assert loaded_model.device == device
    
    # Test forward pass with loaded model
    output = loaded_model(x)
    assert 'loss' in output
    assert 'recon_loss' in output
    assert 'kl_loss' in output
    assert output['recon'].shape == (batch_size, model_params['output_dim'])

def test_model_generation(model_params, device):
    """Test model generation functionality."""
    # Test VAE generation
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    vae_model.eval()
    with torch.no_grad():
        generated = vae_model.generate(num_samples=10)
        assert generated.shape == (10, model_params['output_dim'])
    
    # Test GAN generation
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    gan_model.eval()
    with torch.no_grad():
        generated = gan_model.generate(num_samples=10)
        assert generated.shape == (10, model_params['output_dim'])
    
    # Test GNN generation
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    gnn_model.eval()
    with torch.no_grad():
        # For GNN, we need edge indices
        edge_index = torch.randint(0, model_params['input_dim'], (2, 100), device=device)
        generated = gnn_model.generate(num_nodes=10, edge_index=edge_index)
        assert generated.shape == (10, model_params['output_dim'])

def test_model_parameters(model_params, device):
    """Test model parameter counts and shapes."""
    # Test VAE parameters
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    vae_summary = vae_model.get_model_summary()
    assert vae_summary['total_params'] > 0
    assert vae_summary['trainable_params'] > 0
    
    # Test GAN parameters
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    gan_summary = gan_model.get_model_summary()
    assert gan_summary['total_params'] > 0
    assert gan_summary['trainable_params'] > 0
    
    # Test GNN parameters
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    gnn_summary = gnn_model.get_model_summary()
    assert gnn_summary['total_params'] > 0
    assert gnn_summary['trainable_params'] > 0

@pytest.mark.gpu
def test_model_device_placement(model_params, device):
    """Test model device placement and tensor operations."""
    # Test VAE device placement
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    x = torch.randn(32, model_params['input_dim'], device=device)
    output = vae_model(x)
    assert output['recon'].device == device
    assert output['z'].device == device
    
    # Test GAN device placement
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    z = torch.randn(32, model_params['latent_dim'], device=device)
    output = gan_model(z)
    assert output['fake'].device == device
    
    # Test GNN device placement
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    edge_index = torch.randint(0, model_params['input_dim'], (2, 100), device=device)
    x = torch.randn(model_params['input_dim'], model_params['input_dim'], device=device)
    output = gnn_model(x, edge_index)
    assert output['recon'].device == device
    assert output['z'].device == device

@pytest.mark.slow
def test_model_training(model_params, device, batch_data):
    """Test model training functionality."""
    # Test VAE training
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    vae_model.train()
    optimizer = torch.optim.Adam(vae_model.parameters())
    output = vae_model(batch_data['x'])
    loss = output['loss']
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test GAN training
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    gan_model.train()
    optimizer = torch.optim.Adam(gan_model.parameters())
    output = gan_model.train_step(batch_data['x'])
    loss = output['g_loss'] + output['d_loss']
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test GNN training
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    gnn_model.train()
    optimizer = torch.optim.Adam(gnn_model.parameters())
    output = gnn_model(batch_data['x'], batch_data['edge_index'])
    loss = output['loss']
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_model_input_validation(model_params, device):
    """Test model input validation."""
    # Test VAE input validation
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    with pytest.raises(ValueError):
        vae_model(torch.randn(32, model_params['input_dim'] + 1, device=device))
    
    # Test GAN input validation
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    with pytest.raises(ValueError):
        gan_model(torch.randn(32, model_params['latent_dim'] + 1, device=device))
    
    # Test GNN input validation
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    with pytest.raises(ValueError):
        gnn_model(
            torch.randn(model_params['input_dim'] + 1, model_params['input_dim'], device=device),
            torch.randint(0, model_params['input_dim'], (2, 100), device=device)
        )

@pytest.mark.integration
def test_model_pipeline(model_params, device, batch_data):
    """Test complete model pipeline including generation and evaluation."""
    # Create models
    vae_model = ModelFactory.create_model(
        model_type='vae',
        device=device,
        **model_params
    )
    gan_model = ModelFactory.create_model(
        model_type='gan',
        device=device,
        **model_params
    )
    gnn_model = ModelFactory.create_model(
        model_type='gnn',
        device=device,
        **model_params
    )
    
    # Test generation pipeline
    num_samples = 10
    generated_samples = []
    
    # VAE generation
    vae_model.eval()
    with torch.no_grad():
        vae_samples = vae_model.generate(num_samples=num_samples)
        generated_samples.append(vae_samples)
    
    # GAN generation
    gan_model.eval()
    with torch.no_grad():
        gan_samples = gan_model.generate(num_samples=num_samples)
        generated_samples.append(gan_samples)
    
    # GNN generation
    gnn_model.eval()
    with torch.no_grad():
        gnn_samples = gnn_model.generate(
            num_nodes=num_samples,
            edge_index=batch_data['edge_index']
        )
        generated_samples.append(gnn_samples)
    
    # Validate generated samples
    for samples in generated_samples:
        assert samples.shape[0] == num_samples
        assert samples.shape[1] == model_params['output_dim']
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
        assert samples.device == device

def test_model_initialization_edge_cases():
    """Test model initialization with edge cases."""
    # Test with minimum dimensions
    min_params = {
        'input_dim': 1,
        'hidden_dim': 2,
        'latent_dim': 1,
        'output_dim': 1,
        'dropout': 0.0
    }
    
    for model_type in ['vae', 'gan', 'gnn']:
        model = ModelFactory.create_model(
            model_type=model_type,
            **min_params
        )
        assert model is not None
    
    # Test with maximum dimensions (within reasonable bounds)
    max_params = {
        'input_dim': 1000,
        'hidden_dim': 2048,
        'latent_dim': 512,
        'output_dim': 1000,
        'dropout': 0.5
    }
    
    for model_type in ['vae', 'gan', 'gnn']:
        model = ModelFactory.create_model(
            model_type=model_type,
            **max_params
        )
        assert model is not None

@pytest.mark.slow
def test_model_memory_usage(model_params, device):
    """Test model memory usage and cleanup."""
    import gc
    
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0
    
    initial_memory = get_memory_usage()
    
    # Create and delete models
    for model_type in ['vae', 'gan', 'gnn']:
        model = ModelFactory.create_model(
            model_type=model_type,
            device=device,
            **model_params
        )
        model_memory = get_memory_usage()
        assert model_memory > initial_memory
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = get_memory_usage()
        assert final_memory <= model_memory 