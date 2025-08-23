"""Saving and loading model weights (HDF5 via h5py if present, else .npz fallback).
TensorFlow permitted only for HDF5 I/O if user has it; optional.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    import h5py  # type: ignore
except ImportError:  # optional
    h5py = None


def save_weights_hdf5(path: str, weights: Dict[str, np.ndarray]):
    if h5py is None:
        # fallback to npz with a note extension mismatch is user's choice
        np.savez(path + '.npz', **weights)
        return
    with h5py.File(path, 'w') as f:
        for k, v in weights.items():
            f.create_dataset(k, data=v)


def load_weights_hdf5(path: str) -> Dict[str, np.ndarray]:
    if h5py is None:
        data = np.load(path + '.npz')
        return {k: data[k] for k in data.files}
    with h5py.File(path, 'r') as f:
        return {k: f[k][()] for k in f.keys()}
