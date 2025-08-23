"""Utility helpers."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from .layers import NAME2LAYER


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((labels.size, num_classes), dtype=np.float32)
    y[np.arange(labels.size), labels] = 1
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
