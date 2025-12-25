"""Loss functions with JIT-compiled operations."""
from __future__ import annotations
import numpy as np
from typing import Dict
from . import jax_backend as backend


class Loss:
    """Base class for all loss functions."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss value.
        
        Args:
            y_pred: Predictions
            y_true: Ground truth labels
            
        Returns:
            Scalar loss value
        """
        raise NotImplementedError
    
    def backward(self) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions.
        
        Returns:
            Gradient array
        """
        raise NotImplementedError


class CategoricalCrossentropy(Loss):
    """Cross-entropy loss for multi-class classification.
    
    Supports both integer labels and one-hot encoded labels.
    Uses JIT-compiled numerically stable softmax computation.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.y_true: np.ndarray = None
        self.y_pred: np.ndarray = None
        self.probs: np.ndarray = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss.
        
        Args:
            y_pred: Logits (before softmax), shape (batch, num_classes)
            y_true: Integer labels or one-hot encoded labels
            
        Returns:
            Scalar loss value
        """
        y_pred = backend.asarray(y_pred)
        y_true = backend.asarray(y_true)
        self.y_true = y_true
        self.y_pred = y_pred
        xp = backend.get_array_module()
        
        # Compute softmax in numerically stable way (subtract max)
        probs = y_pred - xp.max(y_pred, axis=1, keepdims=True)
        probs = xp.exp(probs)
        probs = probs / xp.sum(probs, axis=1, keepdims=True)
        self.probs = probs
        
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Integer labels: directly index probabilities
            indices = y_true.reshape(-1)
            log_likelihood = -xp.log(probs[xp.arange(len(y_true)), indices] + 1e-12)
            return float(backend.to_numpy(xp.mean(log_likelihood)))
        else:
            # One-hot labels: compute weighted sum
            return float(backend.to_numpy(-xp.mean(xp.sum(y_true * xp.log(probs + 1e-12), axis=1))))

    def backward(self) -> np.ndarray:
        """Compute gradient of loss w.r.t. logits.
        
        Returns:
            Gradient array with same shape as predictions
        """
        y_true = self.y_true
        probs = self.probs
        xp = backend.get_array_module()
        
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Integer labels
            grad = probs.copy() if hasattr(probs, 'copy') else xp.array(probs)
            indices = y_true.reshape(-1)
            # For JAX, we need to use .at[].set() syntax
            if backend.USE_JAX and backend.jnp is not None:
                grad = grad.at[xp.arange(len(y_true)), indices].add(-1)
            else:
                grad[xp.arange(len(y_true)), indices] -= 1
            grad = grad / len(y_true)
            return grad
        else:
            # One-hot labels
            return (probs - y_true) / y_true.shape[0]


class MSE(Loss):
    """Mean Squared Error loss for regression tasks."""
    
    def __init__(self) -> None:
        super().__init__()
        self.y_pred: np.ndarray = None
        self.y_true: np.ndarray = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE loss.
        
        Args:
            y_pred: Predictions
            y_true: Ground truth values
            
        Returns:
            Scalar loss value
        """
        y_pred = backend.asarray(y_pred)
        y_true = backend.asarray(y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        xp = backend.get_array_module()
        return float(backend.to_numpy(xp.mean((y_pred - y_true) ** 2)))
        
    def backward(self) -> np.ndarray:
        """Compute gradient of MSE w.r.t. predictions.
        
        Returns:
            Gradient array with same shape as predictions
        """
        batch_size: int = self.y_true.size
        return 2 * (self.y_pred - self.y_true) / batch_size


NAME2LOSS: Dict[str, type] = {
    'categorical_crossentropy': CategoricalCrossentropy,
    'cce': CategoricalCrossentropy,
    'mse': MSE
}
