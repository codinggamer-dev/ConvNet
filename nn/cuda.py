"""
CUDA support module using CuPy for GPU acceleration.
Falls back to NumPy when CUDA is not available.
"""
import os
import warnings

# Try importing CuPy for CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    # Check if CUDA devices are actually available
    try:
        cp.cuda.Device(0).use()
        cp.array([1, 2, 3])  # Test allocation
    except Exception:
        CUDA_AVAILABLE = False
        cp = None
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

import numpy as np

# Global flag for CUDA usage (can be disabled via environment variable)
USE_CUDA = CUDA_AVAILABLE and os.environ.get('NN_DISABLE_CUDA', '0') != '1'

def get_array_module(arr=None):
    """
    Get the appropriate array module (cupy or numpy) for the given array.
    If no array is provided, returns the default module based on USE_CUDA.
    """
    if arr is not None and USE_CUDA and cp is not None:
        return cp.get_array_module(arr)
    elif USE_CUDA and cp is not None:
        return cp
    else:
        return np

def to_cpu(arr):
    """Move array to CPU (NumPy)."""
    if USE_CUDA and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def to_gpu(arr):
    """Move array to GPU (CuPy) if CUDA is available."""
    if USE_CUDA and cp is not None and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def asarray(arr):
    """Convert to appropriate array type (GPU if available, CPU otherwise)."""
    if USE_CUDA and cp is not None:
        return cp.asarray(arr)
    return np.asarray(arr)

def zeros_like(arr):
    """Create zeros with same shape and type as input array."""
    xp = get_array_module(arr)
    return xp.zeros_like(arr)

def ones_like(arr):
    """Create ones with same shape and type as input array."""
    xp = get_array_module(arr)
    return xp.ones_like(arr)

def is_cuda_array(arr):
    """Check if array is a CUDA array."""
    return USE_CUDA and cp is not None and isinstance(arr, cp.ndarray)

def get_device_name():
    """Get the name of the current device."""
    if USE_CUDA and cp is not None:
        try:
            device = cp.cuda.Device()
            return f"CUDA:{device.id} ({cp.cuda.get_device_name(device.id)})"
        except Exception:
            return "CUDA (device info unavailable)"
    return "CPU"

def synchronize():
    """Synchronize CUDA operations (no-op on CPU)."""
    if USE_CUDA and cp is not None:
        cp.cuda.Stream.null.synchronize()

# Initialize and print status
if __name__ == "__main__":
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print(f"Using CUDA: {USE_CUDA}")
    print(f"Device: {get_device_name()}")
else:
    # Only show warning if explicitly trying to use CUDA but it's not available
    if os.environ.get('NN_FORCE_CUDA', '0') == '1' and not CUDA_AVAILABLE:
        warnings.warn("CUDA was requested but is not available. Falling back to CPU.")