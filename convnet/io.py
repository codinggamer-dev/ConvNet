"""Saving and loading model weights (HDF5 via h5py if present, else .npz fallback)."""
from __future__ import annotations
import numpy as np
from typing import Dict

try:
    import h5py  # type: ignore
    H5PY_AVAILABLE: bool = True
except ImportError:  # optional
    h5py = None
    H5PY_AVAILABLE = False


def save_weights_hdf5(path: str, weights: Dict[str, np.ndarray]) -> None:
    """Save model weights to HDF5 file (or NPZ if h5py unavailable).
    
    Args:
        path: File path for saving weights
        weights: Dictionary mapping parameter names to arrays
    """
    if h5py is None:
        # Fallback to npz if h5py not available
        npz_path: str = path if path.endswith('.npz') else path + '.npz'
        np.savez(npz_path, **weights)
        return
    
    with h5py.File(path, 'w') as f:
        for k, v in weights.items():
            f.create_dataset(k, data=v, compression='gzip', compression_opts=4)


def load_weights_hdf5(path: str) -> Dict[str, np.ndarray]:
    """Load model weights from HDF5 file (or NPZ if h5py unavailable).
    
    Args:
        path: File path to load weights from
        
    Returns:
        Dictionary mapping parameter names to arrays
        
    Raises:
        FileNotFoundError: If weight file doesn't exist
    """
    if h5py is None:
        # Fallback to npz if h5py not available
        npz_path: str = path if path.endswith('.npz') else path + '.npz'
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}
    
    with h5py.File(path, 'r') as f:
        return {k: np.array(f[k]) for k in f.keys()}
