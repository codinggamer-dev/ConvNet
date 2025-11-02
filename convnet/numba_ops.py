"""Numba-accelerated operations for performance-critical loops.

This module provides JIT-compiled functions using numba for operations
that are difficult to vectorize efficiently with pure NumPy.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable

try:
    from numba import jit, prange
    NUMBA_AVAILABLE: bool = True
except ImportError:
    NUMBA_AVAILABLE: bool = False
    # Fallback decorator that does nothing
    def jit(*args: Any, **kwargs: Any) -> Callable:
        """Fallback decorator when numba is not available."""
        def decorator(func: Callable) -> Callable:
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range  # Use standard range when numba not available


@jit(nopython=True, parallel=True, cache=True)
def col2im_backward_numba(dcols_flat, dx_padded, batch, out_h, out_w, kh, kw, c, stride):
    """Numba-accelerated col2im backward pass for Conv2D.
    
    Args:
        dcols_flat: Gradient columns, shape (batch*out_h*out_w, kh*kw*c)
        dx_padded: Output padded gradient array, shape (batch, h_p, w_p, c)
        batch: Batch size
        out_h: Output height
        out_w: Output width
        kh: Kernel height
        kw: Kernel width
        c: Number of channels
        stride: Stride value
    """
    for n in prange(batch):
        for i in range(out_h):
            for j in range(out_w):
                idx = n * out_h * out_w + i * out_w + j
                i_pos = i * stride
                j_pos = j * stride
                
                for ki in range(kh):
                    for kj in range(kw):
                        for ch in range(c):
                            col_idx = ki * kw * c + kj * c + ch
                            dx_padded[n, i_pos + ki, j_pos + kj, ch] += dcols_flat[idx, col_idx]


@jit(nopython=True, parallel=True, cache=True)
def maxpool_forward_numba(x, y, max_idx, batch, in_h, in_w, out_h, out_w, ph, pw, c, stride):
    """Numba-accelerated forward pass for MaxPool2D.
    
    Args:
        x: Input array, shape (batch, in_h, in_w, c)
        y: Output array, shape (batch, out_h, out_w, c)
        max_idx: Index array to store max positions, shape (batch, out_h, out_w, c)
        batch: Batch size
        in_h, in_w: Input height and width
        out_h, out_w: Output height and width
        ph, pw: Pool height and width
        c: Number of channels
        stride: Stride value
    """
    for n in prange(batch):
        for i in range(out_h):
            for j in range(out_w):
                for ch in range(c):
                    max_val = -np.inf
                    max_index = 0
                    
                    for pi in range(ph):
                        for pj in range(pw):
                            hi = i * stride + pi
                            wj = j * stride + pj
                            if hi < in_h and wj < in_w:
                                val = x[n, hi, wj, ch]
                                if val > max_val:
                                    max_val = val
                                    max_index = pi * pw + pj
                    
                    y[n, i, j, ch] = max_val
                    max_idx[n, i, j, ch] = max_index


@jit(nopython=True, parallel=True, cache=True)
def maxpool_backward_numba(dx, grad, max_idx, batch, out_h, out_w, ph, pw, c, stride):
    """Numba-accelerated backward pass for MaxPool2D.
    
    Args:
        dx: Input gradient array, shape (batch, in_h, in_w, c)
        grad: Output gradient array, shape (batch, out_h, out_w, c)
        max_idx: Index array with max positions, shape (batch, out_h, out_w, c)
        batch: Batch size
        out_h, out_w: Output height and width
        ph, pw: Pool height and width
        c: Number of channels
        stride: Stride value
    """
    for n in prange(batch):
        for i in range(out_h):
            for j in range(out_w):
                for ch in range(c):
                    idx = max_idx[n, i, j, ch]
                    pi = idx // pw
                    pj = idx % pw
                    hi = i * stride + pi
                    wj = j * stride + pj
                    dx[n, hi, wj, ch] += grad[n, i, j, ch]


# Status function
def is_numba_available() -> bool:
    """Check if numba is available and working.
    
    Returns:
        True if numba is successfully imported
    """
    return NUMBA_AVAILABLE
