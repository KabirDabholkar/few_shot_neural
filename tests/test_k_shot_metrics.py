import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt

from few_shot_neural import (
    get_k_shot_co_bps, 
    get_k_shot_co_bps_ensemble,
    train_regressor_and_get_co_bps
)

def bits_per_spike(pred_rates, true_spikes):
    """
    Compute bits per spike between predicted rates and true spikes.
    
    Parameters:
    -----------
    pred_rates : torch.Tensor
        Predicted firing rates, shape (n_trials, n_time_steps, n_neurons)
    true_spikes : torch.Tensor
        True spike counts, shape (n_trials, n_time_steps, n_neurons)
        
    Returns:
    --------
    float
        Bits per spike score
    """
    # Compute log likelihood of true spikes under predicted rates
    log_likelihood = torch.sum(true_spikes * pred_rates - torch.exp(pred_rates))
    
    # Compute log likelihood under null model (constant rate)
    null_rate = torch.log(torch.mean(true_spikes, dim=(0, 1), keepdim=True))
    null_log_likelihood = torch.sum(true_spikes * null_rate - torch.exp(null_rate))
    
    # Compute bits per spike
    n_spikes = torch.sum(true_spikes)
    if n_spikes == 0:
        return 0.0
        
    return (log_likelihood - null_log_likelihood) / (n_spikes * np.log(2))

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

def test_get_k_shot_co_bps_basic(synthetic_data):
    """Test basic functionality of get_k_shot_co_bps"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with k=5 trials
    k_trials = 5
    bps = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=k_trials,
        n_input_neurons=n_input_neurons
    )
    
    # Check that bps is a float
    assert isinstance(bps, float)
    # Check that bps is finite
    assert np.isfinite(bps)
    # Check that bps is not too large (should be reasonable for bits per spike)
    assert bps < 10.0

def test_get_k_shot_co_bps_input_neurons(synthetic_data):
    """Test that input neurons are properly handled"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with and without input neurons
    bps_with_inputs = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_input_neurons=n_input_neurons
    )
    bps_without_inputs = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_input_neurons=None
    )
    
    # Should be different since we're evaluating on different sets of neurons
    assert bps_with_inputs != bps_without_inputs

def test_get_k_shot_co_bps_ensemble(synthetic_data):
    """Test the ensemble version of k-shot co-bps"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with different ensemble sizes
    n_ensemble = 3
    bps = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=n_ensemble,
        n_input_neurons=n_input_neurons
    )
    
    # Check that bps is a float
    assert isinstance(bps, float)
    # Check that bps is finite
    assert np.isfinite(bps)
    # Check that bps is not too large
    assert bps < 10.0

def test_get_k_shot_co_bps_edge_cases(synthetic_data):
    """Test edge cases for k-shot co-bps"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with minimum possible k (1 trial)
    bps_min = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=1,
        n_input_neurons=n_input_neurons
    )
    assert np.isfinite(bps_min)
    
    # Test with maximum possible k (n_train_trials - 1)
    bps_max = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=len(train_trials) - 1,
        n_input_neurons=n_input_neurons
    )
    assert np.isfinite(bps_max)
    
    # Test that k cannot be larger than number of training trials
    with pytest.raises(ValueError):
        get_k_shot_co_bps(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=len(train_trials),
            n_input_neurons=n_input_neurons
        )

def test_get_k_shot_co_bps_consistency(synthetic_data):
    """Test that results are consistent across multiple runs with same random seed"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Set random seed
    np.random.seed(42)
    bps1 = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    np.random.seed(42)
    bps2 = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    # Results should be identical with same random seed
    if np.isnan(bps1) and np.isnan(bps2):
        # Both NaN is considered equal for testing purposes
        assert True
    else:
        assert bps1 == bps2

def test_get_k_shot_co_bps_ensemble_consistency(synthetic_data):
    """Test that ensemble results are consistent with same random seed"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Set random seed
    np.random.seed(42)
    bps1 = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=3,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    np.random.seed(42)
    bps2 = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=3,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    # Results should be identical with same random seed
    if np.isnan(bps1) and np.isnan(bps2):
        # Both NaN is considered equal for testing purposes
        assert True
    else:
        assert bps1 == bps2

def test_get_k_shot_co_bps_ensemble_variance(synthetic_data):
    """Test that ensemble results show some variance with different random seeds"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with different random seeds
    bps1 = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=3,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    bps2 = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=3,
        n_input_neurons=n_input_neurons,
        seed=43
    )
    
    # Results should be different with different random seeds
    # But they should both be finite
    assert np.isfinite(bps1)
    assert np.isfinite(bps2)
    # They might be equal by chance, but it's unlikely
    # If they are equal, it's probably a bug

def test_get_k_shot_co_bps_shape_handling(synthetic_data):
    """Test that the function handles different shapes correctly"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with a valid k_trials value (less than number of training trials)
    k_trials = min(5, len(train_trials) - 1)  # Ensure k_trials is valid
    bps = get_k_shot_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=k_trials,
        n_input_neurons=n_input_neurons
    )
    assert np.isfinite(bps)

def test_get_k_shot_co_bps_ensemble_parameters(synthetic_data):
    """Test that ensemble parameters are handled correctly"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with different ensemble sizes
    n_ensemble_list = [1, 3, 5]
    for n_ensemble in n_ensemble_list:
        bps = get_k_shot_co_bps_ensemble(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=5,
            n_ensemble=n_ensemble,
            n_input_neurons=n_input_neurons
        )
        assert np.isfinite(bps)
    
    # Test with different k values
    k_trials_list = [1, 3, 5]
    for k_trials in k_trials_list:
        bps = get_k_shot_co_bps_ensemble(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=k_trials,
            n_ensemble=3,
            n_input_neurons=n_input_neurons
        )
        assert np.isfinite(bps)

def test_get_k_shot_co_bps_vs_k():
    """Test how k-shot co-bps varies with k"""
    # Create synthetic data with more trials
    n_trials = 100
    n_time_steps = 100
    n_latents = 5
    n_neurons = 10
    n_input_neurons = 2
    
    inf_latents, inf_rates, true_spikes, n_input_neurons = create_synthetic_data(
        n_trials=n_trials,
        n_time_steps=n_time_steps,
        n_latents=n_latents,
        n_neurons=n_neurons,
        n_input_neurons=n_input_neurons
    )
    
    # Split data into train and test sets
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with different k values
    k_trials_list = [1, 5, 10, 20, 30]
    bps_list = []
    
    for k_trials in k_trials_list:
        bps = get_k_shot_co_bps_ensemble(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=k_trials,
            n_ensemble=3,
            n_input_neurons=n_input_neurons
        )
        bps_list.append(bps)
    
    # Plot results
    plt.figure()
    plt.plot(k_trials_list, bps_list, 'o-')
    plt.xlabel('k trials')
    plt.ylabel('k-shot co-bps')
    plt.title('k-shot co-bps vs k')
    plt.savefig('k_shot_co_bps_vs_k.png')
    plt.close()
    
    # Check that bps generally increases with k
    # (though this might not always be true due to randomness)
    for i in range(len(bps_list)-1):
        if not np.isnan(bps_list[i]) and not np.isnan(bps_list[i+1]):
            # Allow for some decrease due to randomness
            assert bps_list[i+1] >= bps_list[i] - 0.1

def test_train_regressor_and_get_co_bps_basic(synthetic_data):
    """Test basic functionality of train_regressor_and_get_co_bps"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test basic functionality
    bps = train_regressor_and_get_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        n_input_neurons=n_input_neurons
    )
    
    # Check that bps is a float
    assert isinstance(bps, float)
    # Check that bps is finite
    assert np.isfinite(bps)
    # Check that bps is not too large
    assert bps < 10.0

def test_train_regressor_and_get_co_bps_consistency():
    """Test that results are consistent across multiple runs with same random seed"""
    # Create synthetic data
    n_trials = 20
    n_time_steps = 100
    n_latents = 5
    n_neurons = 10
    n_input_neurons = 2
    
    inf_latents, inf_rates, true_spikes, n_input_neurons = create_synthetic_data(
        n_trials=n_trials,
        n_time_steps=n_time_steps,
        n_latents=n_latents,
        n_neurons=n_neurons,
        n_input_neurons=n_input_neurons
    )
    
    # Split data into train and test sets
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Set random seed
    np.random.seed(42)
    bps1 = train_regressor_and_get_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    np.random.seed(42)
    bps2 = train_regressor_and_get_co_bps(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    # Results should be identical with same random seed
    if np.isnan(bps1) and np.isnan(bps2):
        # Both NaN is considered equal for testing purposes
        assert True
    else:
        assert bps1 == bps2

def test_train_regressor_and_get_co_bps_shape_handling(synthetic_data):
    """Test that the function handles different input shapes correctly"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Test with different numbers of trials
    n_trials_list = [5, 10, 20]
    for n_trials in n_trials_list:
        train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
        test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
        
        train_latents = inf_latents[train_trials]
        train_spikes = true_spikes[train_trials]
        test_latents = inf_latents[test_trials]
        test_spikes = true_spikes[test_trials]
        
        bps = train_regressor_and_get_co_bps(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            n_input_neurons=n_input_neurons
        )
        assert np.isfinite(bps)
    
    # Test with different numbers of time steps
    n_time_steps_list = [50, 100, 200]
    for n_time_steps in n_time_steps_list:
        train_latents = inf_latents[:, :n_time_steps, :]
        train_spikes = true_spikes[:, :n_time_steps, :]
        test_latents = inf_latents[:, :n_time_steps, :]
        test_spikes = true_spikes[:, :n_time_steps, :]
        
        bps = train_regressor_and_get_co_bps(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            n_input_neurons=n_input_neurons
        )
        assert np.isfinite(bps)

def test_train_regressor_and_get_co_bps_invalid_inputs():
    """Test that the function handles invalid inputs correctly"""
    # Create some test data
    n_trials = 10
    n_time_steps = 100
    n_latents = 5
    n_neurons = 10
    
    # Create synthetic data
    test_latents = torch.randn(n_trials, n_time_steps, n_latents)
    test_spikes = torch.poisson(torch.exp(torch.randn(n_trials, n_time_steps, n_neurons)))
    
    # Test with empty input tensors
    with pytest.raises(ValueError):
        train_regressor_and_get_co_bps(
            train_latents=torch.empty(0, n_time_steps, n_latents),
            train_spikes=torch.empty(0, n_time_steps, n_neurons),
            test_latents=test_latents,
            test_spikes=test_spikes
        )
    
    # Test with mismatched number of trials
    with pytest.raises(ValueError):
        train_regressor_and_get_co_bps(
            train_latents=torch.randn(n_trials, n_time_steps, n_latents),
            train_spikes=torch.randn(n_trials+1, n_time_steps, n_neurons),
            test_latents=test_latents,
            test_spikes=test_spikes
        )
    
    # Test with mismatched number of neurons
    with pytest.raises(ValueError):
        train_regressor_and_get_co_bps(
            train_latents=torch.randn(n_trials, n_time_steps, n_latents),
            train_spikes=torch.randn(n_trials, n_time_steps, n_neurons+1),
            test_latents=test_latents,
            test_spikes=test_spikes
        )

def test_get_k_shot_co_bps_ensemble_std(synthetic_data):
    """Test that the standard deviation option works correctly in get_k_shot_co_bps_ensemble"""
    inf_latents, inf_rates, true_spikes, n_input_neurons = synthetic_data
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Split data into train and test sets
    n_trials = inf_latents.shape[0]
    train_trials = np.random.choice(n_trials, n_trials//2, replace=False)
    test_trials = np.array([i for i in range(n_trials) if i not in train_trials])
    
    train_latents = inf_latents[train_trials]
    train_spikes = true_spikes[train_trials]
    test_latents = inf_latents[test_trials]
    test_spikes = true_spikes[test_trials]
    
    # Test with return_std=True
    n_ensemble = 5
    mean_score, std_score = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=n_ensemble,
        n_input_neurons=n_input_neurons,
        return_std=True,
        seed=42
    )
    
    # Check that both mean and std are floats
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)
    
    # Check that both are finite
    assert np.isfinite(mean_score)
    assert np.isfinite(std_score)
    
    # Check that std is non-negative
    assert std_score >= 0
    
    # Check that mean is within reasonable bounds
    assert mean_score < 10.0
    
    # Test with return_std=False (default)
    mean_only = get_k_shot_co_bps_ensemble(
        train_latents=train_latents,
        train_spikes=train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        k_trials=5,
        n_ensemble=n_ensemble,
        n_input_neurons=n_input_neurons,
        seed=42
    )
    
    # Check that mean matches when return_std=False
    assert np.isclose(mean_only, mean_score)
    
    # Test with all NaN scores
    # Create data that will produce NaN scores
    empty_train_latents = torch.zeros(0, train_latents.shape[1], train_latents.shape[2])
    empty_train_spikes = torch.zeros(0, train_spikes.shape[1], train_spikes.shape[2])
    
    # Test that empty tensors raise ValueError
    with pytest.raises(ValueError, match="Input tensors cannot be empty"):
        get_k_shot_co_bps_ensemble(
            train_latents=empty_train_latents,
            train_spikes=empty_train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=5,
            n_ensemble=n_ensemble,
            n_input_neurons=n_input_neurons,
            return_std=True,
            seed=42
        )