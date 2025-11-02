"""
CUDA support module using CuPy for GPU acceleration.
Falls back to NumPy when CUDA is not available.
"""
from __future__ import annotations
import os
import warnings
from typing import Any, Union, Optional

# Try importing CuPy for CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE: bool = True
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

# Type aliases
ArrayLike = Union[np.ndarray, Any]  # Any to handle CuPy arrays without importing

# Global flag for CUDA usage (can be disabled via environment variable)
USE_CUDA: bool = CUDA_AVAILABLE and os.environ.get('NN_DISABLE_CUDA', '0') != '1'


def get_array_module(arr: Optional[ArrayLike] = None) -> Any:
    """Get the appropriate array module (cupy or numpy) for the given array.
    
    Args:
        arr: Input array (optional)
        
    Returns:
        cupy or numpy module
    """
    if arr is not None and USE_CUDA and cp is not None:
        return cp.get_array_module(arr)
    elif USE_CUDA and cp is not None:
        return cp
    else:
        return np


def to_cpu(arr: ArrayLike) -> np.ndarray:
    """Move array to CPU (NumPy).
    
    Args:
        arr: Input array (NumPy or CuPy)
        
    Returns:
        NumPy array
    """
    if USE_CUDA and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if isinstance(arr, (int, float, bool)):
        return np.array(arr)
    return arr


def to_gpu(arr: ArrayLike) -> ArrayLike:
    """Move array to GPU (CuPy) if CUDA is available.
    
    Args:
        arr: Input array
        
    Returns:
        CuPy array if CUDA available, otherwise NumPy array
    """
    if USE_CUDA and cp is not None and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def asarray(arr: ArrayLike) -> ArrayLike:
    """Convert to appropriate array type (GPU if available, CPU otherwise).
    
    Args:
        arr: Input data
        
    Returns:
        Array on appropriate device
    """
    if USE_CUDA and cp is not None:
        return cp.asarray(arr)
    return np.asarray(arr)


def zeros_like(arr: ArrayLike) -> ArrayLike:
    """Create zeros with same shape and type as input array.
    
    Args:
        arr: Reference array
        
    Returns:
        Zero array with same properties
    """
    xp = get_array_module(arr)
    return xp.zeros_like(arr)


def ones_like(arr: ArrayLike) -> ArrayLike:
    """Create ones with same shape and type as input array.
    
    Args:
        arr: Reference array
        
    Returns:
        Ones array with same properties
    """
    xp = get_array_module(arr)
    return xp.ones_like(arr)


def is_cuda_array(arr: ArrayLike) -> bool:
    """Check if array is a CUDA array.
    
    Args:
        arr: Input array
        
    Returns:
        True if array is on GPU
    """
    return USE_CUDA and cp is not None and isinstance(arr, cp.ndarray)


def is_cupy_available() -> bool:
    """Check if CuPy is available and CUDA is enabled.
    
    Returns:
        True if CuPy and CUDA are available
    """
    return CUDA_AVAILABLE and USE_CUDA


def get_device_name() -> str:
    """Get the name of the current device.
    
    Returns:
        Device name string
    """
    if USE_CUDA and cp is not None:
        try:
            device = cp.cuda.Device()
            return f"CUDA:{device.id} ({cp.cuda.get_device_name(device.id)})"
        except Exception:
            return "CUDA (device info unavailable)"
    return "CPU"


def synchronize() -> None:
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