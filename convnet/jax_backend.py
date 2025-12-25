"""
JAX backend for GPU/TPU acceleration.
Uses NumPy for complex operations and JAX for element-wise and matrix ops.
"""
from __future__ import annotations
import os
from typing import Any, Union, Optional, Callable

# Try importing JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
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
    JAX_AVAILABLE = False
    GPU_AVAILABLE = False
    TPU_AVAILABLE = False
    
    def jit(func: Callable = None, **kwargs) -> Callable:
        if func is None:
            return lambda f: f
        return func

import numpy as np

ArrayLike = Union[np.ndarray, Any]
USE_GPU = GPU_AVAILABLE and os.environ.get('NN_DISABLE_GPU', '0') != '1'
USE_JAX = JAX_AVAILABLE and os.environ.get('NN_DISABLE_JAX', '0') != '1'


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


def get_device_name() -> str:
    """Get device name."""
    if USE_JAX and jax:
        try:
            device = jax.devices()[0]
            if device.platform == 'gpu':
                return f"GPU ({device.device_kind})"
            elif device.platform == 'tpu':
                return "TPU"
        except Exception:
            pass
    return "CPU"


# ============================================================================
# im2col / col2im - NumPy stride tricks (fast, no JIT needed)
# ============================================================================

def im2col(x_padded: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
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
    return cols.reshape(batch * out_h * out_w, kh * kw * c)


def col2im_backward(dcols: np.ndarray, x_shape: tuple, kh: int, kw: int, 
                    stride: int, pad: int) -> np.ndarray:
    """Backward pass for im2col."""
    batch, h, w, c = x_shape
    h_p, w_p = h + 2*pad, w + 2*pad
    out_h = (h_p - kh) // stride + 1
    out_w = (w_p - kw) // stride + 1
    
    dx_padded = np.zeros((batch, h_p, w_p, c), dtype=dcols.dtype)
    dcols_reshaped = dcols.reshape(batch, out_h, out_w, kh, kw, c)
    
    for i in range(out_h):
        for j in range(out_w):
            dx_padded[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :] += dcols_reshaped[:, i, j]
    
    if pad > 0:
        return dx_padded[:, pad:-pad, pad:-pad, :]
    return dx_padded


# ============================================================================
# MaxPool - efficient reshape for stride==pool_size
# ============================================================================

def maxpool_forward(x: np.ndarray, pool_h: int, pool_w: int, stride: int) -> tuple:
    """MaxPool2D forward pass."""
    batch, h, w, c = x.shape
    out_h = (h - pool_h) // stride + 1
    out_w = (w - pool_w) // stride + 1
    
    if stride == pool_h == pool_w and h % pool_h == 0 and w % pool_w == 0:
        x_reshaped = x.reshape(batch, out_h, pool_h, out_w, pool_w, c)
        y = np.max(x_reshaped, axis=(2, 4))
        return y, (x_reshaped, y)
    
    y = np.zeros((batch, out_h, out_w, c), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i*stride:i*stride+pool_h, j*stride:j*stride+pool_w, :]
            y[:, i, j, :] = np.max(patch, axis=(1, 2))
    return y, None


def maxpool_backward(grad: np.ndarray, cache: Any, x: np.ndarray, 
                     pool_h: int, pool_w: int, stride: int) -> np.ndarray:
    """MaxPool2D backward pass."""
    batch, h, w, c = x.shape
    out_h = (h - pool_h) // stride + 1
    out_w = (w - pool_w) // stride + 1
    
    if cache is not None:
        x_reshaped, y = cache
        mask = (x_reshaped == y[:, :, None, :, None, :])
        mask_sum = np.maximum(np.sum(mask, axis=(2, 4), keepdims=True), 1)
        mask = mask / mask_sum
        return (mask * grad[:, :, None, :, None, :]).reshape(batch, h, w, c)
    
    dx = np.zeros_like(x)
    for i in range(out_h):
        for j in range(out_w):
            i0, j0 = i * stride, j * stride
            patch = x[:, i0:i0+pool_h, j0:j0+pool_w, :]
            mask = (patch == np.max(patch, axis=(1, 2), keepdims=True))
            mask = mask / np.maximum(np.sum(mask, axis=(1, 2), keepdims=True), 1)
            dx[:, i0:i0+pool_h, j0:j0+pool_w, :] += mask * grad[:, i:i+1, j:j+1, :]
    return dx


# ============================================================================
# Element-wise operations - JIT compiled
# ============================================================================

if USE_JAX and jnp is not None:
    
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

else:
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


# ============================================================================
# Batch Normalization
# ============================================================================

def batch_norm_forward(x, gamma, beta, running_mean, running_var,
                       momentum=0.9, eps=1e-5, training=True):
    """BatchNorm forward pass."""
    xp = np if isinstance(x, np.ndarray) else jnp
    
    if training:
        mean = xp.mean(x, axis=(0, 1, 2))
        var = xp.var(x, axis=(0, 1, 2))
        x_hat = (x - mean) / xp.sqrt(var + eps)
        new_rm = momentum * running_mean + (1 - momentum) * mean
        new_rv = momentum * running_var + (1 - momentum) * var
        return gamma * x_hat + beta, x_hat, mean, var, new_rm, new_rv
    else:
        x_hat = (x - running_mean) / xp.sqrt(running_var + eps)
        return gamma * x_hat + beta, x_hat, running_mean, running_var, running_mean, running_var


def batch_norm_backward(grad, x, x_hat, gamma, batch_var, eps=1e-5):
    """BatchNorm backward pass."""
    xp = np if isinstance(x, np.ndarray) else jnp
    N = x.shape[0] * x.shape[1] * x.shape[2]
    
    dgamma = xp.sum(grad * x_hat, axis=(0, 1, 2))
    dbeta = xp.sum(grad, axis=(0, 1, 2))
    
    dx_hat = grad * gamma
    std_inv = 1 / xp.sqrt(batch_var + eps)
    batch_mean = xp.mean(x, axis=(0, 1, 2))
    
    dvar = xp.sum(dx_hat * (x - batch_mean) * -0.5 * (batch_var + eps)**(-1.5), axis=(0, 1, 2))
    dmean = xp.sum(dx_hat * -std_inv, axis=(0, 1, 2))
    
    dx = dx_hat * std_inv + dvar * 2 * (x - batch_mean) / N + dmean / N
    return dx, dgamma, dbeta


# Backward compatibility
to_cpu = to_numpy
to_gpu = asarray
to_jax = asarray
