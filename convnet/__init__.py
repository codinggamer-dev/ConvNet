"""convnet - Minimal modular convolutional neural network framework using only numpy+tqdm.

ConvNet-NumPy: A minimal, educational CNN framework built entirely from scratch.

Provides:
- Layer base classes (Conv2D, Dense, Activation, Flatten, MaxPool2D, Dropout, BatchNorm2D)
- Model class with build, forward (predict), backward, train, save, load
- Optimizers (SGD, Adam)
- Losses (categorical crossentropy, mse)
- Dataset utilities for MNIST-like IDX gzip files
- HDF5 weight save/load (via h5py if available or fallback to manual NumPy .npz)
- Simple multi-threaded data loader using ThreadPoolExecutor
- Optional numba JIT compilation for performance-critical loops (Conv2D, MaxPool2D)
- Optional CUDA acceleration via CuPy

Constraints: core math is pure numpy; tqdm allowed for progress bars. Optional h5py for saving/loading .hdf5 only.
Optional numba for JIT compilation, optional cupy for GPU acceleration.
"""
from __future__ import annotations
import os as _os

__version__: str = "1.0.0"
__author__: str = "codinggamer-dev"
__license__: str = "MIT"


def _auto_configure_threads() -> None:
    """Set BLAS / OpenMP thread counts to all available CPU cores if user
    hasn't specified them. Must run before NumPy loads heavy backends.

    Environment vars respected (won't override if already set):
    OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS.
    Disable by setting NN_DISABLE_AUTO_THREADS=1.
    """
    if _os.environ.get('NN_DISABLE_AUTO_THREADS') == '1':
        return
    cores: int = _os.cpu_count() or 1
    for var in [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'
    ]:
        if var not in _os.environ:
            _os.environ[var] = str(cores)

_auto_configure_threads()

from . import layers, losses, optim, data, model, utils, cuda, numba_ops  # noqa: E402
from .model import Model  # noqa: E402
from . import io  # noqa: E402

__all__ = [
    'layers', 'losses', 'optim', 'data', 'model', 'utils', 'io', 'cuda', 'numba_ops', 
    'Model', '_auto_configure_threads', '__version__', '__author__', '__license__'
]
