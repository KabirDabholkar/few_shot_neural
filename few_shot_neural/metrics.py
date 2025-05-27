import numpy as np
import torch
from sklearn.metrics import r2_score
from .poisson_regressor import PoissonRegressor

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

def train_regressor_and_get_co_bps(train_latents, train_spikes, test_latents, test_spikes, n_input_neurons=None, seed=None):
    """
    Train a Poisson regressor on training data and evaluate co-bps on test data.
    
    Parameters:
    -----------
    train_latents : torch.Tensor
        Training latent variables, shape (n_train_trials, n_time_steps, n_latents)
    train_spikes : torch.Tensor
        Training spike counts, shape (n_train_trials, n_time_steps, n_neurons)
    test_latents : torch.Tensor
        Test latent variables, shape (n_test_trials, n_time_steps, n_latents)
    test_spikes : torch.Tensor
        Test spike counts, shape (n_test_trials, n_time_steps, n_neurons)
    n_input_neurons : int, optional
        Number of input neurons. If provided, only evaluates on non-input neurons.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    float
        co-bps score on test data
    """
    # Input validation
    if train_latents.shape[0] == 0 or test_latents.shape[0] == 0:
        raise ValueError("Input tensors cannot be empty")
    
    if train_latents.shape[0] != train_spikes.shape[0]:
        raise ValueError("Number of training trials must match between latents and spikes")
        
    if test_latents.shape[0] != test_spikes.shape[0]:
        raise ValueError("Number of test trials must match between latents and spikes")
    
    # Check number of neurons before any input neuron filtering
    if train_spikes.shape[-1] != test_spikes.shape[-1]:
        raise ValueError("Number of neurons must match between train and test")
    
    # Get shapes
    n_train_trials, n_time_steps, n_latents = train_latents.shape
    n_neurons = train_spikes.shape[-1]
    
    # Prepare data for training
    if n_input_neurons is not None:
        if n_input_neurons >= n_neurons:
            raise ValueError("n_input_neurons must be less than total number of neurons")
        train_spikes = train_spikes[:, :, n_input_neurons:]
        test_spikes = test_spikes[:, :, n_input_neurons:]
        n_neurons = n_neurons - n_input_neurons
    
    # Train Poisson regressor
    reg = PoissonRegressor(max_iter=1000, seed=seed)
    reg.fit(
        train_latents.reshape(-1, n_latents),
        train_spikes.reshape(-1, n_neurons)
    )
    
    # Get predictions
    pred_rates = reg.predict(test_latents.reshape(-1, n_latents))
    
    # Reshape predictions and test data for bits_per_spike
    n_test_trials = test_latents.shape[0]
    pred_rates = pred_rates.reshape(n_test_trials, n_time_steps, n_neurons)
    test_spikes = test_spikes.reshape(n_test_trials, n_time_steps, n_neurons)
    
    # Compute bits per spike
    bps = bits_per_spike(
        torch.tensor(np.log(pred_rates)).float(),
        test_spikes.detach().clone().float()
    ).item()
    
    return bps

def sample_k_trials(latents, spikes, k_trials, seed=None):
    """
    Sample k trials from the given data.
    
    Parameters:
    -----------
    latents : torch.Tensor
        Latent variables, shape (n_trials, n_time_steps, n_latents)
    spikes : torch.Tensor
        Spike counts, shape (n_trials, n_time_steps, n_neurons)
    k_trials : int
        Number of trials to sample
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (sampled_latents, sampled_spikes) where each has k trials
    """
    if k_trials >= latents.shape[0]:
        raise ValueError("k_trials must be less than the number of trials")
        
    if seed is not None:
        np.random.seed(seed)
        
    n_trials = latents.shape[0]
    sampled_indices = np.random.choice(n_trials, k_trials, replace=False)
    
    return latents[sampled_indices], spikes[sampled_indices]

def get_k_shot_co_bps(train_latents, train_spikes, test_latents, test_spikes, k_trials, n_input_neurons=None, seed=None):
    """
    Compute k-shot co-bps by training a Poisson regressor on k trials and evaluating on remaining trials.
    
    Parameters:
    -----------
    train_latents : torch.Tensor
        Training latent variables, shape (n_train_trials, n_time_steps, n_latents)
    train_spikes : torch.Tensor
        Training spike counts, shape (n_train_trials, n_time_steps, n_neurons) 
    test_latents : torch.Tensor
        Test latent variables, shape (n_test_trials, n_time_steps, n_latents)
    test_spikes : torch.Tensor
        Test spike counts, shape (n_test_trials, n_time_steps, n_neurons)
    k_trials : int
        Number of trials to use for training
    n_input_neurons : int, optional
        Number of input neurons. If provided, only evaluates on non-input neurons.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    float
        k-shot co-bps score
    """
    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        
    # Input validation
    if train_latents.shape[0] == 0 or test_latents.shape[0] == 0:
        raise ValueError("Input tensors cannot be empty")
    
    if train_latents.shape[0] != train_spikes.shape[0]:
        raise ValueError("Number of training trials must match between latents and spikes")
        
    if test_latents.shape[0] != test_spikes.shape[0]:
        raise ValueError("Number of test trials must match between latents and spikes")
    
    if train_spikes.shape[-1] != test_spikes.shape[-1]:
        raise ValueError("Number of neurons must match between train and test")
    
    # Sample k trials from training data
    k_train_latents, k_train_spikes = sample_k_trials(
        train_latents, 
        train_spikes, 
        k_trials, 
        seed=seed
    )
    
    # Use the new function to compute co-bps
    return train_regressor_and_get_co_bps(
        train_latents=k_train_latents,
        train_spikes=k_train_spikes,
        test_latents=test_latents,
        test_spikes=test_spikes,
        n_input_neurons=n_input_neurons,
        seed=seed
    )

def get_k_shot_co_bps_ensemble(train_latents, train_spikes, test_latents, test_spikes, k_trials, n_ensemble=5, n_input_neurons=None, seed=None, return_std=False):
    """
    Compute k-shot co-bps multiple times with different random k-trial selections and average the results.
    
    Parameters:
    -----------
    train_latents : torch.Tensor
        Training latent variables, shape (n_train_trials, n_time_steps, n_latents)
    train_spikes : torch.Tensor
        Training spike counts, shape (n_train_trials, n_time_steps, n_neurons)
    test_latents : torch.Tensor
        Test latent variables, shape (n_test_trials, n_time_steps, n_latents)
    test_spikes : torch.Tensor
        Test spike counts, shape (n_test_trials, n_time_steps, n_neurons)
    k_trials : int
        Number of trials to use for training
    n_ensemble : int
        Number of times to repeat the k-shot evaluation
    n_input_neurons : int, optional
        Number of input neurons
    seed : int, optional
        Random seed for reproducibility
    return_std : bool, optional
        If True, return both mean and standard deviation of scores
        
    Returns:
    --------
    float or tuple
        If return_std is False, returns average k-shot co-bps score across ensemble
        If return_std is True, returns (mean, std) of k-shot co-bps scores
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    scores = []
    for i in range(n_ensemble):
        # Use different seeds for each ensemble run
        current_seed = seed + i if seed is not None else None
        score = get_k_shot_co_bps(
            train_latents=train_latents,
            train_spikes=train_spikes,
            test_latents=test_latents,
            test_spikes=test_spikes,
            k_trials=k_trials,
            n_input_neurons=n_input_neurons,
            seed=current_seed
        )
        if np.isfinite(score):  # Only include finite scores
            scores.append(score)
    
    if not scores:  # If all scores were NaN/infinite
        return (np.nan, np.nan) if return_std else np.nan
        
    if return_std:
        return np.mean(scores), np.std(scores)
    return np.mean(scores) 