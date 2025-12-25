"""Optimizers with JAX acceleration support."""
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, Optional, Dict
from . import jax_backend as backend


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"learning rate must be positive, got {lr}")
        self.lr: float = lr
        self.weight_decay: float = 0.0
        self.clip_norm: Optional[float] = None
        
    def configure(self, weight_decay: float = 0.0, 
                  clip_norm: Optional[float] = None) -> 'Optimizer':
        """Configure regularization parameters.
        
        Args:
            weight_decay: L2 regularization coefficient
            clip_norm: Maximum gradient norm for clipping
            
        Returns:
            Self for method chaining
        """
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        return self
        
    def _apply_regularization(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Apply weight decay and gradient clipping.
        
        Args:
            p: Parameter array
            g: Gradient array
            
        Returns:
            Regularized gradient
        """
        xp = backend.get_array_module()
        
        # L2 regularization (weight decay)
        if self.weight_decay > 0:
            g = g + self.weight_decay * p
        
        # Gradient clipping by norm
        if self.clip_norm is not None:
            norm = float(backend.to_numpy(xp.linalg.norm(g)))
            if norm > self.clip_norm:
                g = g * (self.clip_norm / norm)
        return g
        
    def step(self, params_and_grads: Iterable[Tuple[Any, str, np.ndarray, np.ndarray]]) -> None:
        """Perform one optimization step.
        
        Args:
            params_and_grads: Iterable of (layer, param_name, parameter, gradient) tuples
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Reset optimizer state (e.g., momentum)."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional momentum."""
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum: float = momentum
        self.v: Dict[int, np.ndarray] = {}
        
    def step(self, params_and_grads: Iterable[Tuple[Any, str, np.ndarray, np.ndarray]]) -> None:
        """Update parameters using SGD with optional momentum.
        
        Args:
            params_and_grads: Iterable of (layer, param_name, parameter, gradient) tuples
        """
        xp = backend.get_array_module()
        for i, (layer, name, p, g) in enumerate(params_and_grads):
            g = self._apply_regularization(p, g)
            if self.momentum > 0:
                # Momentum: accumulate velocity
                v = self.v.get(i, xp.zeros_like(g))
                v = self.momentum * v - self.lr * g
                self.v[i] = v
                # Update using layer's update method for JAX compatibility
                p_new = p + v
                layer.update_param(name, p_new)
            else:
                # Standard SGD - update using layer's method for JAX compatibility
                p_new = p - self.lr * g
                layer.update_param(name, p_new)


class Adam(Optimizer):
    """Adam optimizer with bias correction and adaptive learning rates."""
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__(lr)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: int = 0
        
    def step(self, params_and_grads: Iterable[Tuple[Any, str, np.ndarray, np.ndarray]]) -> None:
        """Update parameters using Adam optimization.
        
        Args:
            params_and_grads: Iterable of (layer, param_name, parameter, gradient) tuples
        """
        xp = backend.get_array_module()
        self.t += 1
        for i, (layer, name, p, g) in enumerate(params_and_grads):
            g = self._apply_regularization(p, g)
            
            # Get or initialize moment estimates
            m = self.m.get(i, xp.zeros_like(g))
            v = self.v.get(i, xp.zeros_like(g))
            
            # Update biased first and second moment estimates
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters using layer's update method for JAX compatibility
            update = self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
            p_new = p - update
            layer.update_param(name, p_new)
            
            # Store updated moments
            self.m[i] = m
            self.v[i] = v


NAME2OPT: Dict[str, type] = {'sgd': SGD, 'adam': Adam}
