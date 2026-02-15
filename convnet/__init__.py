"""convnet - Educational CNN framework with optional JAX/SciPy acceleration.

A minimal, clean CNN framework built from scratch for learning deep learning.
Supports optional JAX for GPU/TPU acceleration or SciPy for CPU optimization.

Quick Start:
    from convnet import Model, Dense, Conv2D, Activation, MaxPool2D, Flatten
    
    model = Model([
        Conv2D(8, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Flatten(), Dense(10), Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', lr=0.001)
"""
from __future__ import annotations
import os as _os

__version__: str = "2.5.1"
__author__: str = "codinggamer-dev"
__license__: str = "MIT"


def _auto_configure_threads() -> None:
    """Auto-configure thread counts for optimal performance.
    
    Supports Intel MKL, OpenBLAS, BLIS, and other BLAS libraries.
    Intel MKL is recommended for Intel CPUs (i3, i5, i7, Xeon).
    """
    if _os.environ.get('NN_DISABLE_AUTO_THREADS') == '1':
        return
    cores: int = _os.cpu_count() or 1
    # Set threading for various BLAS implementations
    # MKL (Intel Math Kernel Library) - best for Intel CPUs
    # OpenBLAS - good open-source alternative
    # BLIS - AMD-optimized BLAS
    for var in ['MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OMP_NUM_THREADS', 
                'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if var not in _os.environ:
            if var == 'MKL_DYNAMIC':
                _os.environ[var] = 'FALSE'  # Disable dynamic threading for consistent performance
            else:
                _os.environ[var] = str(cores)

_auto_configure_threads()

# Core imports
from .model import Model  # noqa: E402
from .layers import (  # noqa: E402
    Dense, Conv2D, MaxPool2D, Flatten, Activation, Dropout, BatchNorm2D
)
from .data import Dataset  # noqa: E402
from . import losses, optim, jax_backend  # noqa: E402

# Convenience aliases
Adam = optim.Adam
SGD = optim.SGD

__all__ = [
    # Main classes
    'Model', 'Dataset',
    # Layers
    'Dense', 'Conv2D', 'MaxPool2D', 'Flatten', 'Activation', 'Dropout', 'BatchNorm2D',
    # Optimizers
    'Adam', 'SGD',
    # Submodules
    'losses', 'optim', 'jax_backend',
]
