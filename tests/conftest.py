import numpy as np
import pytest
import torch

def create_synthetic_data(n_trials=20, n_time_steps=100, n_latents=5, n_neurons=10, n_input_neurons=2, scale=0.1):
    """Helper function to create synthetic data for testing"""
    # Create synthetic latent variables with smaller values to prevent overflow
    inf_latents = torch.randn(n_trials, n_time_steps, n_latents) * scale
    
    # Create synthetic rates that depend on latents
    # Use a simple linear mapping with some noise
    W = torch.randn(n_latents, n_neurons) * scale
    b = torch.randn(n_neurons) * scale
    inf_rates = torch.exp(torch.matmul(inf_latents, W) + b)
    
    # Generate true spikes from rates using Poisson distribution
    true_spikes = torch.poisson(inf_rates)
    
    return inf_latents, inf_rates, true_spikes, n_input_neurons

@pytest.fixture
def synthetic_data():
    """Fixture to provide synthetic data for tests"""
    return create_synthetic_data()

@pytest.fixture
def synthetic_data_large():
    """Fixture to provide larger synthetic data for tests"""
    return create_synthetic_data(n_trials=100, n_time_steps=200, n_latents=10, n_neurons=20, n_input_neurons=5) 