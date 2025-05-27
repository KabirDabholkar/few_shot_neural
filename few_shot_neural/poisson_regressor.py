import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional

class PoissonRegressor:
    """
    A Poisson regression model implemented in PyTorch with a scikit-learn-like interface.
    Uses maximum likelihood estimation to fit the model.
    """
    
    def __init__(self, 
                 max_iter: int = 1000,
                 learning_rate: float = 0.01,
                 tol: float = 1e-4,
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the Poisson regressor.
        
        Parameters:
        -----------
        max_iter : int, default=1000
            Maximum number of iterations for optimization
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        tol : float, default=1e-4
            Tolerance for optimization convergence
        device : str, optional
            Device to use for computation ('cpu' or 'cuda')
        seed : int, optional
            Random seed for reproducibility. If None, no seed is set.
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = None
        self.bias = None
        
        # Create a dedicated random number generator
        self.rng = np.random.RandomState(seed)
        self.torch_generator = torch.Generator(device=self.device)
        if seed is not None:
            self.torch_generator.manual_seed(seed)
        
    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to torch tensor if needed and ensure it's detached."""
        if isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.float32, device=self.device)
        return X.detach().to(self.device)
    
    def _initialize_parameters(self, n_features: int, n_outputs: int):
        """Initialize model parameters."""
        # Use the dedicated RNG for initialization
        weights = torch.tensor(
            self.rng.randn(n_features, n_outputs) * 0.01,
            dtype=torch.float32,
            device=self.device
        )
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(
            torch.zeros(1, n_outputs, device=self.device)
        )
        
    def _compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss."""
        log_rates = X @ self.weights + self.bias
        rates = torch.exp(log_rates)
        return -torch.mean(y * log_rates - rates - torch.lgamma(y + 1))
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> 'PoissonRegressor':
        """
        Fit the Poisson regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, n_outputs)
            Target values (count data)
            
        Returns:
        --------
        self : PoissonRegressor
            The fitted model
        """
        # Convert and detach inputs
        X = self._to_tensor(X)
        y = self._to_tensor(y)
        
        if self.weights is None:
            self._initialize_parameters(X.shape[1], y.shape[1])
            
        optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
        prev_loss = float('inf')
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            loss = self._compute_loss(X, y)
            loss.backward()
            optimizer.step()
            
            if abs(prev_loss - loss.item()) < self.tol:
                break
            prev_loss = loss.item()
            
        return self
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict using the fitted model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples, n_outputs)
            Predicted rates
        """
        X = self._to_tensor(X)
        with torch.no_grad():
            log_rates = X @ self.weights + self.bias
            rates = torch.exp(log_rates)
        return rates.cpu().numpy()
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights.detach().cpu().numpy(),
            'bias': self.bias.detach().cpu().numpy()
        } 