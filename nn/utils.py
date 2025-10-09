"""Utility helpers."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Union
from .layers import NAME2LAYER
from . import cuda

# Type alias for arrays that could be NumPy or CuPy
ArrayLike = Union[np.ndarray, Any]  # Any to avoid cupy import


def one_hot(labels: ArrayLike, num_classes: int) -> ArrayLike:
    """Convert integer labels to one-hot encoding, supporting both CPU and GPU arrays."""
    xp = cuda.get_array_module(labels)
    y = xp.zeros((labels.size, num_classes), dtype=xp.float32)
    y[xp.arange(labels.size), labels] = 1
    return y


def serialize_layers(layers) -> List[Dict[str, Any]]:
    conf = []
    for layer in layers:
        conf.append(layer.to_config())
    return conf


def deserialize_layers(config_list):
    layers = []
    for conf in config_list:
        cls = NAME2LAYER[conf['class']]
        layers.append(cls(**conf['config']))
    return layers
