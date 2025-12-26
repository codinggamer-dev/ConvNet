"""Numba-accelerated operations for maximum CPU performance.

This module provides JIT-compiled parallel functions for operations
that are difficult to vectorize efficiently with pure NumPy.
All functions use parallel=True and fastmath=True for maximum speed.

Note: cache=False to avoid Numba cache corruption issues that can cause segfaults.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

try:
    from numba import jit, prange, get_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback - these won't be used if Numba isn't available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    def get_num_threads():
        return 1


# ============================================================================
# im2col operations - Critical for Conv2D forward
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def im2col_numba(x_padded: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """Extract patches for convolution using parallel loops.
    
    Args:
        x_padded: Padded input (batch, h_p, w_p, c)
        kh, kw: Kernel dimensions
        stride: Stride value
        
    Returns:
        cols: Column matrix (batch * out_h * out_w, kh * kw * c)
    """
    batch, h_p, w_p, c = x_padded.shape
    out_h = (h_p - kh) // stride + 1
    out_w = (w_p - kw) // stride + 1
    patch_size = kh * kw * c
    
    cols = np.empty((batch * out_h * out_w, patch_size), dtype=x_padded.dtype)
    
    # Parallel over output positions for better load balancing
    total_positions = batch * out_h * out_w
    for pos in prange(total_positions):
        n = pos // (out_h * out_w)
        rem = pos % (out_h * out_w)
        i = rem // out_w
        j = rem % out_w
        
        i_start = i * stride
        j_start = j * stride
        
        col_idx = 0
        for ki in range(kh):
            for kj in range(kw):
                for ch in range(c):
                    cols[pos, col_idx] = x_padded[n, i_start + ki, j_start + kj, ch]
                    col_idx += 1
    
    return cols


# ============================================================================
# col2im operations - Critical for Conv2D backward
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def col2im_backward_numba(dcols: np.ndarray, batch: int, h: int, w: int, c: int,
                          kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Scatter gradients back to image space with parallel accumulation.
    
    Uses kernel-major iteration to avoid race conditions within a batch.
    Pre-computes output dimensions to avoid potential issues with integer calculations.
    
    Args:
        dcols: Gradient columns (batch * out_h * out_w, kh * kw * c)
        batch, h, w, c: Original input dimensions
        kh, kw: Kernel dimensions
        stride: Stride value
        pad: Padding value
        
    Returns:
        dx: Input gradient (batch, h, w, c)
    """
    # Pre-compute dimensions as local variables
    h_p = h + 2 * pad
    w_p = w + 2 * pad
    out_h = (h_p - kh) // stride + 1
    out_w = (w_p - kw) // stride + 1
    hw_out = out_h * out_w
    
    # Allocate output
    dx_padded = np.zeros((batch, h_p, w_p, c), dtype=dcols.dtype)
    
    # Reshape dcols for easier indexing: (batch, out_h, out_w, kh * kw * c)
    dcols_4d = dcols.reshape(batch, out_h, out_w, kh * kw * c)
    
    # Parallel over batch dimension (each batch element is independent)
    for n in prange(batch):
        for i in range(out_h):
            i_start = i * stride
            for j in range(out_w):
                j_start = j * stride
                
                # Iterate over kernel and channels
                col_idx = 0
                for ki in range(kh):
                    for kj in range(kw):
                        for ch in range(c):
                            dx_padded[n, i_start + ki, j_start + kj, ch] += dcols_4d[n, i, j, col_idx]
                            col_idx += 1
    
    # Remove padding
    if pad > 0:
        dx = np.empty((batch, h, w, c), dtype=dcols.dtype)
        for n in prange(batch):
            for i in range(h):
                for j in range(w):
                    for ch in range(c):
                        dx[n, i, j, ch] = dx_padded[n, pad + i, pad + j, ch]
        return dx
    return dx_padded


@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def col2im_backward_numba_strided(dcols: np.ndarray, batch: int, h: int, w: int, c: int,
                                   kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Optimized col2im for stride=1 case using kernel-major iteration.
    
    For stride=1, we can iterate over kernel positions and use vectorized adds.
    """
    h_p = h + 2 * pad
    w_p = w + 2 * pad
    out_h = (h_p - kh) // stride + 1
    out_w = (w_p - kw) // stride + 1
    
    # Reshape dcols to (batch, out_h, out_w, kh, kw, c)
    dcols_reshaped = dcols.reshape(batch, out_h, out_w, kh, kw, c)
    dx_padded = np.zeros((batch, h_p, w_p, c), dtype=dcols.dtype)
    
    # Iterate over kernel positions (small loop, e.g., 9 for 3x3)
    for ki in range(kh):
        for kj in range(kw):
            # Parallel over batch
            for n in prange(batch):
                for i in range(out_h):
                    i_dst = i * stride + ki
                    for j in range(out_w):
                        j_dst = j * stride + kj
                        for ch in range(c):
                            dx_padded[n, i_dst, j_dst, ch] += dcols_reshaped[n, i, j, ki, kj, ch]
    
    if pad > 0:
        dx = np.empty((batch, h, w, c), dtype=dcols.dtype)
        for n in prange(batch):
            for i in range(h):
                for j in range(w):
                    for ch in range(c):
                        dx[n, i, j, ch] = dx_padded[n, pad + i, pad + j, ch]
        return dx
    return dx_padded


# ============================================================================
# MaxPool operations - Optimized forward and backward
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def maxpool_forward_numba(x: np.ndarray, pool_h: int, pool_w: int, 
                          stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """MaxPool forward with parallel argmax computation.
    
    Parallelizes over all output positions for maximum utilization.
    """
    batch, h, w, c = x.shape
    out_h = (h - pool_h) // stride + 1
    out_w = (w - pool_w) // stride + 1
    
    y = np.empty((batch, out_h, out_w, c), dtype=x.dtype)
    max_indices = np.empty((batch, out_h, out_w, c), dtype=np.int32)
    
    # Flatten iteration for better parallel utilization
    total = batch * out_h * out_w
    for pos in prange(total):
        n = pos // (out_h * out_w)
        rem = pos % (out_h * out_w)
        i = rem // out_w
        j = rem % out_w
        
        i_start = i * stride
        j_start = j * stride
        
        for ch in range(c):
            max_val = x[n, i_start, j_start, ch]
            max_idx = 0
            
            for pi in range(pool_h):
                for pj in range(pool_w):
                    val = x[n, i_start + pi, j_start + pj, ch]
                    if val > max_val:
                        max_val = val
                        max_idx = pi * pool_w + pj
            
            y[n, i, j, ch] = max_val
            max_indices[n, i, j, ch] = max_idx
    
    return y, max_indices


@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def maxpool_backward_numba(grad: np.ndarray, max_indices: np.ndarray,
                           batch: int, h: int, w: int, c: int,
                           pool_h: int, pool_w: int, stride: int) -> np.ndarray:
    """MaxPool backward with parallel scatter.
    
    Parallelizes over all output positions for maximum utilization.
    """
    out_h = grad.shape[1]
    out_w = grad.shape[2]
    
    dx = np.zeros((batch, h, w, c), dtype=grad.dtype)
    
    # Parallel over batch (to avoid race conditions on dx)
    for n in prange(batch):
        for i in range(out_h):
            for j in range(out_w):
                i_start = i * stride
                j_start = j * stride
                
                for ch in range(c):
                    max_idx = max_indices[n, i, j, ch]
                    pi = max_idx // pool_w
                    pj = max_idx % pool_w
                    dx[n, i_start + pi, j_start + pj, ch] += grad[n, i, j, ch]
    
    return dx


# ============================================================================
# ReLU operations - Simple but benefits from parallelization on large arrays
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def relu_forward_numba(x: np.ndarray) -> np.ndarray:
    """ReLU forward with parallel computation."""
    out = np.empty_like(x)
    x_flat = x.ravel()
    out_flat = out.ravel()
    
    for i in prange(x_flat.size):
        out_flat[i] = max(0.0, x_flat[i])
    
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def relu_backward_numba(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """ReLU backward with parallel computation."""
    dx = np.empty_like(grad)
    x_flat = x.ravel()
    grad_flat = grad.ravel()
    dx_flat = dx.ravel()
    
    for i in prange(x_flat.size):
        dx_flat[i] = grad_flat[i] if x_flat[i] > 0 else 0.0
    
    return dx


# ============================================================================
# Dense layer operations - Optimized for small matrices
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def dense_backward_dw_numba(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Compute dW = x.T @ grad with parallel outer loop."""
    # x: (batch, in_features), grad: (batch, out_features)
    batch, in_features = x.shape
    out_features = grad.shape[1]
    
    dW = np.zeros((in_features, out_features), dtype=x.dtype)
    
    # Parallel over input features
    for i in prange(in_features):
        for j in range(out_features):
            acc = 0.0
            for b in range(batch):
                acc += x[b, i] * grad[b, j]
            dW[i, j] = acc
    
    return dW


# ============================================================================
# Batch Normalization operations
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def batchnorm_forward_numba(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                            eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BatchNorm forward with parallel computation."""
    # x: (batch, h, w, c) or (batch, features)
    if x.ndim == 4:
        batch, h, w, c = x.shape
        n = batch * h * w
    else:
        batch, c = x.shape
        n = batch
        h = w = 1
    
    # Compute mean
    mean = np.zeros(c, dtype=x.dtype)
    for ch in prange(c):
        acc = 0.0
        if x.ndim == 4:
            for b in range(batch):
                for i in range(h):
                    for j in range(w):
                        acc += x[b, i, j, ch]
        else:
            for b in range(batch):
                acc += x[b, ch]
        mean[ch] = acc / n
    
    # Compute variance
    var = np.zeros(c, dtype=x.dtype)
    for ch in prange(c):
        acc = 0.0
        if x.ndim == 4:
            for b in range(batch):
                for i in range(h):
                    for j in range(w):
                        diff = x[b, i, j, ch] - mean[ch]
                        acc += diff * diff
        else:
            for b in range(batch):
                diff = x[b, ch] - mean[ch]
                acc += diff * diff
        var[ch] = acc / n
    
    # Normalize and scale
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = np.empty_like(x)
    out = np.empty_like(x)
    
    if x.ndim == 4:
        for b in prange(batch):
            for i in range(h):
                for j in range(w):
                    for ch in range(c):
                        x_hat[b, i, j, ch] = (x[b, i, j, ch] - mean[ch]) * std_inv[ch]
                        out[b, i, j, ch] = gamma[ch] * x_hat[b, i, j, ch] + beta[ch]
    else:
        for b in prange(batch):
            for ch in range(c):
                x_hat[b, ch] = (x[b, ch] - mean[ch]) * std_inv[ch]
                out[b, ch] = gamma[ch] * x_hat[b, ch] + beta[ch]
    
    return out, x_hat, mean, var


# ============================================================================
# Softmax with cross-entropy (fused for numerical stability)
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def softmax_numba(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax with parallel computation."""
    batch, classes = x.shape
    out = np.empty_like(x)
    
    for b in prange(batch):
        # Find max for numerical stability
        max_val = x[b, 0]
        for c in range(1, classes):
            if x[b, c] > max_val:
                max_val = x[b, c]
        
        # Compute exp and sum
        exp_sum = 0.0
        for c in range(classes):
            out[b, c] = np.exp(x[b, c] - max_val)
            exp_sum += out[b, c]
        
        # Normalize
        for c in range(classes):
            out[b, c] /= exp_sum
    
    return out


# ============================================================================
# Dropout operations
# ============================================================================

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def dropout_forward_numba(x: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Dropout forward with parallel computation."""
    out = np.empty_like(x)
    x_flat = x.ravel()
    mask_flat = mask.ravel()
    out_flat = out.ravel()
    
    for i in prange(x_flat.size):
        out_flat[i] = x_flat[i] * mask_flat[i] * scale
    
    return out


@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def dropout_backward_numba(grad: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Dropout backward with parallel computation."""
    dx = np.empty_like(grad)
    grad_flat = grad.ravel()
    mask_flat = mask.ravel()
    dx_flat = dx.ravel()
    
    for i in prange(grad_flat.size):
        dx_flat[i] = grad_flat[i] * mask_flat[i] * scale
    
    return dx


# ============================================================================
# Utility functions
# ============================================================================

def is_numba_available() -> bool:
    """Check if Numba is available."""
    return NUMBA_AVAILABLE


def get_numba_threads() -> int:
    """Get number of Numba threads."""
    if NUMBA_AVAILABLE:
        return get_num_threads()
    return 1


def warmup_numba():
    """Warmup Numba JIT compilation with small arrays."""
    if not NUMBA_AVAILABLE:
        return
    
    # Small dummy arrays to trigger compilation
    # Use batch=4, h=8, w=8, c=2, kernel=3x3, pad=1
    # With pad=1: h_p=10, w_p=10, out_h=(10-3)//1+1=8, out_w=8
    x = np.random.randn(4, 10, 10, 2).astype(np.float32)  # Already padded
    
    # Warmup im2col: output is (batch*out_h*out_w, kh*kw*c) = (4*8*8, 3*3*2) = (256, 18)
    cols = im2col_numba(x, 3, 3, 1)
    
    # Warmup col2im: dcols shape must match (batch*out_h*out_w, kh*kw*c)
    dcols = np.random.randn(4 * 8 * 8, 3 * 3 * 2).astype(np.float32)
    _ = col2im_backward_numba(dcols, 4, 8, 8, 2, 3, 3, 1, 1)
    
    # Warmup maxpool
    _, idx = maxpool_forward_numba(x, 2, 2, 2)
    _ = maxpool_backward_numba(np.random.randn(4, 4, 4, 2).astype(np.float32),
                                idx, 4, 8, 8, 2, 2, 2, 2)
    
    # Warmup relu
    _ = relu_forward_numba(x)
    _ = relu_backward_numba(x, x)
    
    # Warmup softmax
    logits = np.random.randn(4, 10).astype(np.float32)
    _ = softmax_numba(logits)
