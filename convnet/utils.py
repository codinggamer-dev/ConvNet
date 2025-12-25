"""Utility helpers."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Union
from .layers import NAME2LAYER, Layer
from . import jax_backend as backend

# Type alias for arrays that could be NumPy or JAX
ArrayLike = Union[np.ndarray, Any]


def one_hot(labels: ArrayLike, num_classes: int) -> ArrayLike:
    """Convert integer labels to one-hot encoding.
    
    Args:
        labels: Integer label array
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded array of shape (num_samples, num_classes)
    """
    xp = backend.get_array_module()
    labels_flat = xp.reshape(labels, (-1,))
    y = xp.zeros((labels_flat.size, num_classes), dtype=xp.float32)
    if backend.USE_JAX and backend.jnp is not None:
        y = y.at[xp.arange(labels_flat.size), labels_flat].set(1)
    else:
        y[xp.arange(labels_flat.size), labels_flat] = 1
    return y


def serialize_layers(layers: List[Layer]) -> List[Dict[str, Any]]:
    """Serialize a list of layers to configuration dictionaries.
    
    Args:
        layers: List of layer instances
        
    Returns:
        List of configuration dictionaries
    """
    return [layer.to_config() for layer in layers]


def deserialize_layers(config_list: List[Dict[str, Any]]) -> List[Layer]:
    """Deserialize layers from configuration dictionaries.
    
    Args:
        config_list: List of configuration dictionaries
        
    Returns:
        List of layer instances
        
    Raises:
        KeyError: If layer class name is not found
    """
    layers: List[Layer] = []
    for conf in config_list:
        cls = NAME2LAYER[conf['class']]
        config = conf['config'].copy()
        
        # Convert lists to tuples for parameters that require tuples
        # (JSON/HDF5 serialization converts tuples to lists)
        if 'kernel_size' in config and isinstance(config['kernel_size'], list):
            config['kernel_size'] = tuple(config['kernel_size'])
        if 'pool_size' in config and isinstance(config['pool_size'], list):
            config['pool_size'] = tuple(config['pool_size'])
        
        layers.append(cls(**config))
    return layers


def get_acceleration_info() -> Dict[str, bool]:
    """Get information about available acceleration backends.
    
    Returns:
        Dict with keys 'jax' and 'gpu' indicating availability
    """
    return {
        'jax': backend.is_jax_available(),
        'gpu': backend.is_gpu_available()
    }
