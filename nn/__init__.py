"""nn - Minimal modular convolutional neural network framework using only numpy+tqdm.

Provides:
- Layer base classes (Conv2D, Dense, Activation, Flatten, MaxPool2D, Dropout, BatchNorm2D)
- Model class with build, forward (predict), backward, train, save, load
- Optimizers (SGD, Adam)
- Losses (categorical crossentropy, mse)
- Dataset utilities for MNIST-like IDX gzip files
- HDF5 weight save/load (via h5py if available or fallback to manual NumPy .npz) with optional TensorFlow interoperability (only for HDF5 I/O section).
- Simple multi-threaded data loader using ThreadPoolExecutor

Constraints: core math is pure numpy; tqdm allowed for progress bars. Optional h5py or tensorflow for saving/loading .hdf5 only.
"""
import os as _os

def _auto_configure_threads():
    """Set BLAS / OpenMP thread counts to all available CPU cores if user
    hasn't specified them. Must run before NumPy loads heavy backends.

    Environment vars respected (won't override if already set):
    OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS.
    Disable by setting NN_DISABLE_AUTO_THREADS=1.
    """
    if _os.environ.get('NN_DISABLE_AUTO_THREADS') == '1':
        return
    cores = _os.cpu_count() or 1
    for var in [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'
    ]:
        if var not in _os.environ:
            _os.environ[var] = str(cores)

_auto_configure_threads()

from . import layers, losses, optim, data, model, utils, cuda  # noqa: E402
from .model import Model  # noqa: E402
from . import io  # noqa: E402

__all__ = [
    'layers', 'losses', 'optim', 'data', 'model', 'utils', 'io', 'cuda', 'Model', '_auto_configure_threads'
]
