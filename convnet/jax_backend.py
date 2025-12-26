"""
Backend for CNN operations with optimal performance:
- JAX for GPU/TPU (XLA acceleration)
- Numba for CPU (parallel JIT compilation)
- Falls back to NumPy if neither is available

All operations stay on-device during computation.
"""
from __future__ import annotations
import os
from typing import Any, Union, Optional, Callable
from functools import partial

import numpy as np

# ============================================================================
# Try importing JAX for GPU/TPU
# ============================================================================
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax
    JAX_AVAILABLE = True
    
    try:
        devices = jax.devices()
        GPU_AVAILABLE = any(d.platform == 'gpu' for d in devices)
        TPU_AVAILABLE = any(d.platform == 'tpu' for d in devices)
    except Exception:
        GPU_AVAILABLE = False
        TPU_AVAILABLE = False
        
except ImportError:
    jax = None
    jnp = None
    lax = None
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    TPU_AVAILABLE = False
    
    def jit(func: Callable = None, **kwargs) -> Callable:
        if func is None:
            return lambda f: f
        return func

# ============================================================================
# Try importing Numba for CPU
# ============================================================================
try:
    from .numba_ops import (
        NUMBA_AVAILABLE,
        im2col_numba,
        col2im_backward_numba,
        col2im_backward_numba_strided,
        maxpool_forward_numba,
        maxpool_backward_numba,
        relu_forward_numba,
        relu_backward_numba,
        softmax_numba,
        warmup_numba,
        get_numba_threads,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    im2col_numba = None
    col2im_backward_numba = None
    col2im_backward_numba_strided = None
    maxpool_forward_numba = None
    maxpool_backward_numba = None
    relu_forward_numba = None
    relu_backward_numba = None
    softmax_numba = None
    warmup_numba = None
    get_numba_threads = lambda: 1


ArrayLike = Union[np.ndarray, Any]

# Determine which backend to use
USE_GPU = GPU_AVAILABLE and os.environ.get('NN_DISABLE_GPU', '0') != '1'
USE_TPU = TPU_AVAILABLE and os.environ.get('NN_DISABLE_TPU', '0') != '1'
USE_JAX = (USE_GPU or USE_TPU) and JAX_AVAILABLE and os.environ.get('NN_DISABLE_JAX', '0') != '1'
USE_NUMBA = NUMBA_AVAILABLE and os.environ.get('NN_DISABLE_NUMBA', '0') != '1' and not USE_JAX

# Warmup Numba JIT on import if using CPU
_NUMBA_WARMED_UP = False


def _ensure_numba_warmup():
    """Warmup Numba JIT compilation (only once)."""
    global _NUMBA_WARMED_UP
    if USE_NUMBA and not _NUMBA_WARMED_UP and warmup_numba:
        warmup_numba()
        _NUMBA_WARMED_UP = True


def get_array_module() -> Any:
    """Get jax.numpy or numpy module."""
    return jnp if USE_JAX and jnp else np


def to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert to NumPy array."""
    if hasattr(arr, 'block_until_ready'):
        arr.block_until_ready()
    return np.asarray(arr)


def asarray(arr: ArrayLike) -> ArrayLike:
    """Convert to appropriate array type."""
    if USE_JAX and jnp:
        return jnp.asarray(arr)
    return np.asarray(arr)


def zeros_like(arr: ArrayLike) -> ArrayLike:
    """Create zeros with same shape."""
    return get_array_module().zeros_like(arr)


def is_jax_available() -> bool:
    return JAX_AVAILABLE and USE_JAX


def is_gpu_available() -> bool:
    return GPU_AVAILABLE and USE_GPU


def is_numba_available() -> bool:
    return USE_NUMBA


def get_device_name() -> str:
    """Get device name."""
    if USE_JAX and jax:
        try:
            device = jax.devices()[0]
            if device.platform == 'gpu':
                return f"GPU ({device.device_kind})"
            elif device.platform == 'tpu':
                return "TPU"
            return "CPU (JAX)"
        except Exception:
            pass
    if USE_NUMBA:
        return f"CPU (Numba {get_numba_threads()}T)"
    return "CPU (NumPy)"


# ============================================================================
# JAX implementations for GPU/TPU
# ============================================================================

if USE_JAX and jnp is not None:
    
    def im2col(x_padded: ArrayLike, kh: int, kw: int, stride: int) -> ArrayLike:
        """Extract patches - JAX native, stays on GPU."""
        batch, h_p, w_p, c = x_padded.shape
        out_h = (h_p - kh) // stride + 1
        out_w = (w_p - kw) // stride + 1
        
        # Use lax.conv_general_dilated_patches for efficient patch extraction
        patches = lax.conv_general_dilated_patches(
            x_padded,
            filter_shape=(kh, kw),
            window_strides=(stride, stride),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        return patches.reshape(batch * out_h * out_w, kh * kw * c)
    
    
    def col2im_backward(dcols: ArrayLike, x_shape: tuple, kh: int, kw: int, 
                        stride: int, pad: int) -> ArrayLike:
        """Backward pass for im2col - JAX native."""
        xp = jnp
        batch, h, w, c = x_shape
        h_p, w_p = h + 2*pad, w + 2*pad
        out_h = (h_p - kh) // stride + 1
        out_w = (w_p - kw) // stride + 1
        
        dcols_reshaped = dcols.reshape(batch, out_h, out_w, kh, kw, c)
        dx_padded = xp.zeros((batch, h_p, w_p, c), dtype=dcols.dtype)
        
        for ki in range(kh):
            for kj in range(kw):
                dg = dcols_reshaped[:, :, :, ki, kj, :]
                if stride == 1:
                    dx_padded = dx_padded.at[:, ki:ki+out_h, kj:kj+out_w, :].add(dg)
                else:
                    for i in range(out_h):
                        for j in range(out_w):
                            dx_padded = dx_padded.at[:, i*stride+ki, j*stride+kj, :].add(dg[:, i, j, :])
        
        if pad > 0:
            return dx_padded[:, pad:-pad, pad:-pad, :]
        return dx_padded
    
    
    def maxpool_forward(x: ArrayLike, pool_h: int, pool_w: int, stride: int) -> tuple:
        """MaxPool2D forward - JAX native using lax.reduce_window."""
        xp = jnp
        batch, h, w, c = x.shape
        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1
        
        # Fast path: use reshape when stride == pool_size
        if stride == pool_h == pool_w and h % pool_h == 0 and w % pool_w == 0:
            x_reshaped = x.reshape(batch, out_h, pool_h, out_w, pool_w, c)
            y = xp.max(x_reshaped, axis=(2, 4))
            return y, (x_reshaped, y)
        
        # General case: use lax.reduce_window (XLA-optimized)
        init_val = float('-inf')
        y = lax.reduce_window(
            x, init_val, lax.max,
            window_dimensions=(1, pool_h, pool_w, 1),
            window_strides=(1, stride, stride, 1),
            padding='VALID'
        )
        return y, None
    
    
    def maxpool_backward(grad: ArrayLike, cache: Any, x: ArrayLike, 
                         pool_h: int, pool_w: int, stride: int) -> ArrayLike:
        """MaxPool2D backward - JAX native."""
        xp = jnp
        batch, h, w, c = x.shape
        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1
        
        # Fast path
        if cache is not None:
            x_reshaped, y = cache
            mask = (x_reshaped == y[:, :, None, :, None, :])
            mask_sum = xp.maximum(xp.sum(mask, axis=(2, 4), keepdims=True), 1)
            mask = mask / mask_sum
            return (mask * grad[:, :, None, :, None, :]).reshape(batch, h, w, c)
        
        # General case: build mask and distribute gradients
        dx = xp.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                i0, j0 = i * stride, j * stride
                patch = lax.dynamic_slice(x, (0, i0, j0, 0), (batch, pool_h, pool_w, c))
                max_val = xp.max(patch, axis=(1, 2), keepdims=True)
                mask = (patch == max_val).astype(x.dtype)
                mask = mask / xp.maximum(xp.sum(mask, axis=(1, 2), keepdims=True), 1)
                g = grad[:, i:i+1, j:j+1, :]
                dx = dx.at[:, i0:i0+pool_h, j0:j0+pool_w, :].add(mask * g)
        return dx


    # JIT-compiled element-wise operations
    @jit
    def relu_forward(x: ArrayLike) -> ArrayLike:
        return jnp.maximum(0, x)
    
    @jit
    def relu_backward(grad: ArrayLike, x: ArrayLike) -> ArrayLike:
        return grad * (x > 0)
    
    @jit
    def sigmoid_forward(x: ArrayLike) -> ArrayLike:
        return 1 / (1 + jnp.exp(-x))
    
    @jit
    def sigmoid_backward(grad: ArrayLike, x: ArrayLike) -> ArrayLike:
        s = 1 / (1 + jnp.exp(-x))
        return grad * s * (1 - s)
    
    @jit
    def tanh_forward(x: ArrayLike) -> ArrayLike:
        return jnp.tanh(x)
    
    @jit
    def tanh_backward(grad: ArrayLike, x: ArrayLike) -> ArrayLike:
        return grad * (1 - jnp.tanh(x)**2)
    
    @jit
    def softmax_forward(x: ArrayLike) -> ArrayLike:
        e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
        return e / jnp.sum(e, axis=-1, keepdims=True)
    
    @jit
    def dense_forward(x: ArrayLike, W: ArrayLike, b: Optional[ArrayLike]) -> ArrayLike:
        y = x @ W
        return y + b if b is not None else y
    
    @jit
    def dense_backward(grad: ArrayLike, x: ArrayLike, W: ArrayLike) -> tuple:
        x_flat = x.reshape(-1, x.shape[-1])
        grad_flat = grad.reshape(-1, grad.shape[-1])
        dW = x_flat.T @ grad_flat
        db = jnp.sum(grad, axis=tuple(range(len(grad.shape)-1)))
        dx = grad @ W.T
        return dx, dW, db
    
    @jit
    def softmax_cross_entropy(logits: ArrayLike, labels: ArrayLike) -> tuple:
        shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
        probs = jnp.exp(shifted) / jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)
        loss = -jnp.mean(jnp.sum(labels * jnp.log(probs + 1e-12), axis=-1))
        return loss, probs
    
    
    def batch_norm_forward(x, gamma, beta, running_mean, running_var,
                           momentum=0.9, eps=1e-5, training=True):
        """BatchNorm forward pass."""
        if training:
            mean = jnp.mean(x, axis=(0, 1, 2))
            var = jnp.var(x, axis=(0, 1, 2))
            x_hat = (x - mean) / jnp.sqrt(var + eps)
            new_rm = momentum * running_mean + (1 - momentum) * mean
            new_rv = momentum * running_var + (1 - momentum) * var
            return gamma * x_hat + beta, x_hat, mean, var, new_rm, new_rv
        else:
            x_hat = (x - running_mean) / jnp.sqrt(running_var + eps)
            return gamma * x_hat + beta, x_hat, running_mean, running_var, running_mean, running_var
    
    
    def batch_norm_backward(grad, x, x_hat, gamma, batch_var, eps=1e-5):
        """BatchNorm backward pass."""
        N = x.shape[0] * x.shape[1] * x.shape[2]
        dgamma = jnp.sum(grad * x_hat, axis=(0, 1, 2))
        dbeta = jnp.sum(grad, axis=(0, 1, 2))
        dx_hat = grad * gamma
        std_inv = 1 / jnp.sqrt(batch_var + eps)
        batch_mean = jnp.mean(x, axis=(0, 1, 2))
        dvar = jnp.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + eps)**(-1.5), axis=(0, 1, 2))
        dmean = jnp.sum(dx_hat * -std_inv, axis=(0, 1, 2))
        dx = dx_hat * std_inv + dvar * 2 * (x - batch_mean) / N + dmean / N
        return dx, dgamma, dbeta


elif USE_NUMBA:
    # ========================================================================
    # Numba implementations for CPU (parallel JIT-compiled)
    # ========================================================================
    
    def im2col(x_padded, kh, kw, stride):
        """im2col using NumPy stride tricks (consistently fastest).
        
        NumPy stride tricks with ascontiguousarray is faster than or equal to
        Numba for all tensor sizes due to highly optimized copy routines.
        """
        _ensure_numba_warmup()
        batch, h_p, w_p, c = x_padded.shape
        out_h = (h_p - kh) // stride + 1
        out_w = (w_p - kw) // stride + 1
        
        # Use stride tricks to create a view, then make contiguous
        cols = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=(batch, out_h, out_w, kh, kw, c),
            strides=(x_padded.strides[0], stride * x_padded.strides[1],
                     stride * x_padded.strides[2], x_padded.strides[1],
                     x_padded.strides[2], x_padded.strides[3])
        )
        return np.ascontiguousarray(cols.reshape(batch * out_h * out_w, kh * kw * c))
    
    
    def col2im_backward(dcols, x_shape, kh, kw, stride, pad):
        """col2im backward using Numba parallel JIT."""
        batch, h, w, c = x_shape
        dcols_c = np.ascontiguousarray(dcols, dtype=np.float32)
        return col2im_backward_numba(dcols_c, batch, h, w, c, kh, kw, stride, pad)
    
    
    def maxpool_forward(x, pool_h, pool_w, stride):
        """MaxPool forward using Numba parallel JIT."""
        _ensure_numba_warmup()
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        y, max_indices = maxpool_forward_numba(x_c, pool_h, pool_w, stride)
        return y, max_indices
    
    
    def maxpool_backward(grad, cache, x, pool_h, pool_w, stride):
        """MaxPool backward using Numba parallel JIT."""
        batch, h, w, c = x.shape
        grad_c = np.ascontiguousarray(grad, dtype=np.float32)
        max_indices = cache  # cache is max_indices from forward
        return maxpool_backward_numba(grad_c, max_indices, batch, h, w, c, pool_h, pool_w, stride)
    
    
    def relu_forward(x):
        """ReLU forward - use Numba for large arrays."""
        if x.size > 10000:
            return relu_forward_numba(np.ascontiguousarray(x, dtype=np.float32))
        return np.maximum(0, x)
    
    
    def relu_backward(grad, x):
        """ReLU backward - use Numba for large arrays."""
        if x.size > 10000:
            return relu_backward_numba(
                np.ascontiguousarray(x, dtype=np.float32),
                np.ascontiguousarray(grad, dtype=np.float32)
            )
        return grad * (x > 0)
    
    
    def sigmoid_forward(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    
    def sigmoid_backward(grad, x):
        s = sigmoid_forward(x)
        return grad * s * (1 - s)
    
    
    def tanh_forward(x):
        return np.tanh(x)
    
    
    def tanh_backward(grad, x):
        return grad * (1 - np.tanh(x)**2)
    
    
    def softmax_forward(x):
        """Softmax - use Numba for 2D inputs."""
        if x.ndim == 2:
            return softmax_numba(np.ascontiguousarray(x, dtype=np.float32))
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    
    
    def dense_forward(x, W, b=None):
        """Dense forward - optimized matmul."""
        x_flat = x.reshape(-1, x.shape[-1])
        y = x_flat @ W
        if b is not None:
            y = y + b
        return y.reshape(x.shape[:-1] + (W.shape[-1],)) if x.ndim > 2 else y
    
    
    def dense_backward(grad, x, W):
        """Dense backward - BLAS-optimized."""
        x_flat = np.ascontiguousarray(x.reshape(-1, x.shape[-1]))
        grad_flat = np.ascontiguousarray(grad.reshape(-1, grad.shape[-1]))
        
        # Use BLAS-friendly operations (column-major access patterns)
        dW = x_flat.T @ grad_flat
        db = np.sum(grad, axis=tuple(range(len(grad.shape)-1)))
        dx = grad @ W.T
        return dx, dW, db
    
    
    def softmax_cross_entropy(logits, labels):
        """Softmax cross-entropy loss."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        loss = -np.mean(np.sum(labels * np.log(probs + 1e-12), axis=-1))
        return loss, probs
    
    
    def batch_norm_forward(x, gamma, beta, running_mean, running_var,
                           momentum=0.9, eps=1e-5, training=True):
        """BatchNorm forward pass."""
        if training:
            mean = np.mean(x, axis=(0, 1, 2))
            var = np.var(x, axis=(0, 1, 2))
            x_hat = (x - mean) / np.sqrt(var + eps)
            new_rm = momentum * running_mean + (1 - momentum) * mean
            new_rv = momentum * running_var + (1 - momentum) * var
            return gamma * x_hat + beta, x_hat, mean, var, new_rm, new_rv
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            return gamma * x_hat + beta, x_hat, running_mean, running_var, running_mean, running_var
    
    
    def batch_norm_backward(grad, x, x_hat, gamma, batch_var, eps=1e-5):
        """BatchNorm backward pass."""
        N = x.shape[0] * x.shape[1] * x.shape[2]
        dgamma = np.sum(grad * x_hat, axis=(0, 1, 2))
        dbeta = np.sum(grad, axis=(0, 1, 2))
        dx_hat = grad * gamma
        std_inv = 1 / np.sqrt(batch_var + eps)
        batch_mean = np.mean(x, axis=(0, 1, 2))
        dvar = np.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + eps)**(-1.5), axis=(0, 1, 2))
        dmean = np.sum(dx_hat * -std_inv, axis=(0, 1, 2))
        dx = dx_hat * std_inv + dvar * 2 * (x - batch_mean) / N + dmean / N
        return dx, dgamma, dbeta


else:
    # ========================================================================
    # Pure NumPy fallback (slowest, but always available)
    # ========================================================================
    
    def im2col(x_padded, kh, kw, stride):
        """Extract patches using NumPy stride tricks."""
        batch, h_p, w_p, c = x_padded.shape
        out_h = (h_p - kh) // stride + 1
        out_w = (w_p - kw) // stride + 1
        cols = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=(batch, out_h, out_w, kh, kw, c),
            strides=(x_padded.strides[0], stride * x_padded.strides[1],
                     stride * x_padded.strides[2], x_padded.strides[1],
                     x_padded.strides[2], x_padded.strides[3])
        )
        return np.ascontiguousarray(cols.reshape(batch * out_h * out_w, kh * kw * c))
    
    
    def col2im_backward(dcols, x_shape, kh, kw, stride, pad):
        """Scatter-add gradients back to image space."""
        batch, h, w, c = x_shape
        h_p, w_p = h + 2*pad, w + 2*pad
        out_h = (h_p - kh) // stride + 1
        out_w = (w_p - kw) // stride + 1
        
        dcols_reshaped = dcols.reshape(batch, out_h, out_w, kh, kw, c)
        dcols_t = np.ascontiguousarray(dcols_reshaped.transpose(0, 3, 4, 1, 2, 5))
        
        dx_padded = np.zeros((batch, h_p, w_p, c), dtype=dcols.dtype)
        
        for ki in range(kh):
            for kj in range(kw):
                dx_padded[:, ki:ki+out_h, kj:kj+out_w, :] += dcols_t[:, ki, kj]
        
        if pad > 0:
            return dx_padded[:, pad:-pad, pad:-pad, :]
        return dx_padded
    
    
    def maxpool_forward(x, pool_h, pool_w, stride):
        """MaxPool forward using vectorized operations."""
        batch, h, w, c = x.shape
        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1
        
        if stride == pool_h == pool_w and h % pool_h == 0 and w % pool_w == 0:
            x_reshaped = x.reshape(batch, out_h, pool_h, out_w, pool_w, c)
            y = np.max(x_reshaped, axis=(2, 4))
            return y, (x_reshaped, y)
        
        cols = np.lib.stride_tricks.as_strided(
            x,
            shape=(batch, out_h, out_w, pool_h, pool_w, c),
            strides=(x.strides[0], stride * x.strides[1], stride * x.strides[2],
                     x.strides[1], x.strides[2], x.strides[3])
        )
        y = np.max(cols, axis=(3, 4))
        return y, (np.ascontiguousarray(cols), y)
    
    
    def maxpool_backward(grad, cache, x, pool_h, pool_w, stride):
        """MaxPool backward using argmax scatter."""
        batch, h, w, c = x.shape
        out_h = (h - pool_h) // stride + 1
        out_w = (w - pool_w) // stride + 1
        
        if cache is not None:
            x_windows, y = cache
            
            if x_windows.shape[2] == pool_h and x_windows.shape[4] == pool_w:
                ph, pw = pool_h, pool_w
                x_transposed = x_windows.transpose(0, 1, 3, 2, 4, 5)
                x_flat = x_transposed.reshape(batch, out_h, out_w, ph * pw, c)
                max_idx = np.argmax(x_flat, axis=3, keepdims=True)
                
                dx_flat = np.zeros((batch, out_h, out_w, ph * pw, c), dtype=grad.dtype)
                np.put_along_axis(dx_flat, max_idx, grad[:, :, :, None, :], axis=3)
                
                return dx_flat.reshape(batch, out_h, out_w, ph, pw, c).transpose(0, 1, 3, 2, 4, 5).reshape(batch, h, w, c)
            else:
                ph, pw = pool_h, pool_w
                x_flat = x_windows.reshape(batch, out_h, out_w, ph * pw, c)
                max_idx = np.argmax(x_flat, axis=3, keepdims=True)
                
                dwindows = np.zeros((batch, out_h, out_w, ph * pw, c), dtype=grad.dtype)
                np.put_along_axis(dwindows, max_idx, grad[:, :, :, None, :], axis=3)
                dwindows = dwindows.reshape(batch, out_h, out_w, ph, pw, c)
                
                dx = np.zeros_like(x)
                for i in range(out_h):
                    for j in range(out_w):
                        i0, j0 = i * stride, j * stride
                        dx[:, i0:i0+pool_h, j0:j0+pool_w, :] += dwindows[:, i, j, :, :, :]
                return dx
        
        dx = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                i0, j0 = i * stride, j * stride
                patch = x[:, i0:i0+pool_h, j0:j0+pool_w, :]
                mask = (patch == np.max(patch, axis=(1, 2), keepdims=True))
                mask = mask / np.maximum(np.sum(mask, axis=(1, 2), keepdims=True), 1)
                dx[:, i0:i0+pool_h, j0:j0+pool_w, :] += mask * grad[:, i:i+1, j:j+1, :]
        return dx
    
    
    def relu_forward(x):
        return np.maximum(0, x)
    
    
    def relu_backward(grad, x):
        return grad * (x > 0)
    
    
    def sigmoid_forward(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    
    def sigmoid_backward(grad, x):
        s = sigmoid_forward(x)
        return grad * s * (1 - s)
    
    
    def tanh_forward(x):
        return np.tanh(x)
    
    
    def tanh_backward(grad, x):
        return grad * (1 - np.tanh(x)**2)
    
    
    def softmax_forward(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    
    
    def dense_forward(x, W, b=None):
        y = x @ W
        return y + b if b is not None else y
    
    
    def dense_backward(grad, x, W):
        x_flat = x.reshape(-1, x.shape[-1])
        grad_flat = grad.reshape(-1, grad.shape[-1])
        dW = x_flat.T @ grad_flat
        db = np.sum(grad, axis=tuple(range(len(grad.shape)-1)))
        dx = grad @ W.T
        return dx, dW, db
    
    
    def softmax_cross_entropy(logits, labels):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        loss = -np.mean(np.sum(labels * np.log(probs + 1e-12), axis=-1))
        return loss, probs
    
    
    def batch_norm_forward(x, gamma, beta, running_mean, running_var,
                           momentum=0.9, eps=1e-5, training=True):
        if training:
            mean = np.mean(x, axis=(0, 1, 2))
            var = np.var(x, axis=(0, 1, 2))
            x_hat = (x - mean) / np.sqrt(var + eps)
            new_rm = momentum * running_mean + (1 - momentum) * mean
            new_rv = momentum * running_var + (1 - momentum) * var
            return gamma * x_hat + beta, x_hat, mean, var, new_rm, new_rv
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            return gamma * x_hat + beta, x_hat, running_mean, running_var, running_mean, running_var
    
    
    def batch_norm_backward(grad, x, x_hat, gamma, batch_var, eps=1e-5):
        N = x.shape[0] * x.shape[1] * x.shape[2]
        dgamma = np.sum(grad * x_hat, axis=(0, 1, 2))
        dbeta = np.sum(grad, axis=(0, 1, 2))
        dx_hat = grad * gamma
        std_inv = 1 / np.sqrt(batch_var + eps)
        batch_mean = np.mean(x, axis=(0, 1, 2))
        dvar = np.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + eps)**(-1.5), axis=(0, 1, 2))
        dmean = np.sum(dx_hat * -std_inv, axis=(0, 1, 2))
        dx = dx_hat * std_inv + dvar * 2 * (x - batch_mean) / N + dmean / N
        return dx, dgamma, dbeta


# Backward compatibility
to_cpu = to_numpy
to_gpu = asarray
to_jax = asarray
