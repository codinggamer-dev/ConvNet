"""Layer definitions for the nn module.
Pure NumPy implementations of common layers with optional CUDA support.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Iterator
from . import cuda
from . import numba_ops

# Helper weight initializer functions

def glorot_uniform(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Glorot/Xavier uniform initialization for stable training.
    
    Args:
        shape: Shape tuple for the weight tensor
        rng: NumPy random generator for reproducibility
        
    Returns:
        Initialized weight array on appropriate device (CPU/GPU)
    """
    fan_in: int = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    fan_out: int = shape[0]
    limit: float = np.sqrt(6.0 / (fan_in + fan_out))
    weights: np.ndarray = rng.uniform(-limit, limit, size=shape).astype(np.float32)
    return cuda.asarray(weights)

class Layer:
    """Abstract layer base class with default implementations."""
    
    def __init__(self) -> None:
        self.built: bool = False
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.trainable: bool = True
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize layer parameters based on input shape."""
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the layer. Default: pass through unchanged."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the layer. Default: pass gradient unchanged."""
        return grad

    def get_params_and_grads(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (parameter, gradient) tuples for optimizer updates."""
        if self.trainable:
            for k, v in self.params.items():
                yield v, self.grads[k]

    def to_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        return {'class': self.__class__.__name__, 'config': {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Layer':
        """Reconstruct layer from configuration."""
        return cls(**config)

class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, units: int, use_bias: bool = True, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        self.units: int = units
        self.use_bias: bool = use_bias
        self.rng: np.random.Generator = rng or np.random.default_rng()
        self.last_x: Optional[np.ndarray] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        in_features = input_shape[-1]
        self.params['W'] = glorot_uniform((in_features, self.units), self.rng)
        if self.use_bias:
            self.params['b'] = cuda.asarray(np.zeros((self.units,), dtype=np.float32))
        self.grads['W'] = cuda.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = cuda.zeros_like(self.params['b'])
        self.output_shape = (*input_shape[:-1], self.units)
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = cuda.asarray(x)
        self.last_x = x
        y = x @ self.params['W']
        if self.use_bias:
            y = y + self.params['b']
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.last_x
        xp = cuda.get_array_module(x)
        self.grads['W'][...] = x.reshape(-1, x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        if self.use_bias:
            self.grads['b'][...] = xp.sum(grad, axis=tuple(range(len(grad.shape)-1)))
        return grad @ self.params['W'].T

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Dense', 'config': {'units': self.units, 'use_bias': self.use_bias}}

class Flatten(Layer):
    """Flatten layer to convert multi-dimensional features to 1D."""
    
    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self.orig_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        if len(input_shape) <= 2:
            self.output_shape = input_shape
        else:
            flat_dim = 1
            for d in input_shape[1:]:
                flat_dim *= d
            self.output_shape = (input_shape[0], flat_dim)
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.orig_shape)

class Activation(Layer):
    """Activation function layer."""
    
    def __init__(self, func: str = 'relu') -> None:
        super().__init__()
        valid_funcs = ['relu', 'sigmoid', 'tanh', 'softmax']
        if func not in valid_funcs:
            raise ValueError(f"Unknown activation '{func}'. Valid options: {valid_funcs}")
        self.func: str = func
        self.trainable: bool = False
        self.last_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = cuda.asarray(x)
        self.last_x = x
        xp = cuda.get_array_module(x)
        
        if self.func == 'relu':
            return xp.maximum(0, x)
        elif self.func == 'sigmoid':
            return 1 / (1 + xp.exp(-x))
        elif self.func == 'tanh':
            return xp.tanh(x)
        elif self.func == 'softmax':
            e = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
            return e / xp.sum(e, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation {self.func}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.last_x
        xp = cuda.get_array_module(x)
        
        if self.func == 'relu':
            return grad * (x > 0)
        elif self.func == 'sigmoid':
            s = 1 / (1 + xp.exp(-x))
            return grad * s * (1 - s)
        elif self.func == 'tanh':
            t = xp.tanh(x)
            return grad * (1 - t**2)
        elif self.func == 'softmax':
            return grad  # Combined with cross-entropy at loss
        else:
            return grad

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Activation', 'config': {'func': self.func}}

class Dropout(Layer):
    """Dropout layer for regularization during training."""
    
    def __init__(self, rate: float = 0.5, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate: float = rate
        self.rng: np.random.Generator = rng or np.random.default_rng()
        self.trainable: bool = False
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = cuda.asarray(x)
        if training:
            mask = (self.rng.random(x.shape) >= self.rate).astype(x.dtype)
            self.mask = cuda.asarray(mask)
            return x * self.mask / (1 - self.rate)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask / (1 - self.rate)

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Dropout', 'config': {'rate': self.rate}}

class Conv2D(Layer):
    """Optimized 2D convolution layer using im2col + GEMM (matrix multiply).

    This replaces the earlier naive nested-loop implementation. The heavy lifting
    is delegated to NumPy's optimized BLAS which can leverage multiple cores.
    """
    def __init__(self, filters: int, kernel_size: Tuple[int, int] = (3,3), stride: int = 1, padding: str = 'same', use_bias: bool = True, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be a tuple of 2 integers, got {kernel_size}")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"kernel_size dimensions must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if padding not in ['same', 'valid']:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        
        self.filters: int = filters
        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: int = stride
        self.padding: str = padding
        self.use_bias: bool = use_bias
        self.rng: np.random.Generator = rng or np.random.default_rng()
        self.last_x: Optional[np.ndarray] = None
        self.cache: Optional[Tuple] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        _, h, w, c = input_shape
        kh, kw = self.kernel_size
        self.params['W'] = glorot_uniform((kh, kw, c, self.filters), self.rng)
        if self.use_bias:
            self.params['b'] = cuda.asarray(np.zeros((self.filters,), dtype=np.float32))
        self.grads['W'] = cuda.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = cuda.zeros_like(self.params['b'])
        
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.stride))
            out_w = int(np.ceil(w / self.stride))
        else:
            out_h = (h - kh) // self.stride + 1
            out_w = (w - kw) // self.stride + 1
        self.output_shape = (None, out_h, out_w, self.filters)
        self.built = True

    def _compute_padding(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Compute padding amounts for 'same' or 'valid' padding mode.
        
        Args:
            h: Input height
            w: Input width
            
        Returns:
            Tuple of (pad_top, pad_bottom, pad_left, pad_right)
        """
        if self.padding == 'same':
            kh, kw = self.kernel_size
            pad_h_total: float = max((np.ceil(h / self.stride) - 1) * self.stride + kh - h, 0.0)
            pad_w_total: float = max((np.ceil(w / self.stride) - 1) * self.stride + kw - w, 0.0)
            pad_top: int = int(pad_h_total // 2)
            pad_bottom: int = int(pad_h_total - pad_top)
            pad_left: int = int(pad_w_total // 2)
            pad_right: int = int(pad_w_total - pad_left)
            return pad_top, pad_bottom, pad_left, pad_right
        return 0, 0, 0, 0

    def _im2col(self, x: np.ndarray) -> Tuple[np.ndarray, int, int, Tuple[int, int, int, int], Tuple[int, ...]]:
        """Convert image to column matrix for efficient convolution."""
        batch, h, w, c = x.shape
        kh, kw = self.kernel_size
        pt, pb, pl, pr = self._compute_padding(h, w)
        xp = cuda.get_array_module(x)
        
        # Pad the input
        x_p = xp.pad(x, ((0,0),(pt,pb),(pl,pr),(0,0)), mode='constant')
        h_p, w_p = x_p.shape[1], x_p.shape[2]
        out_h = (h_p - kh)//self.stride + 1
        out_w = (w_p - kw)//self.stride + 1
        
        # Extract patches using stride tricks (works with CuPy too)
        cols = xp.lib.stride_tricks.as_strided(
            x_p,
            shape=(batch, out_h, out_w, kh, kw, c),
            strides=(x_p.strides[0], self.stride*x_p.strides[1], self.stride*x_p.strides[2], x_p.strides[1], x_p.strides[2], x_p.strides[3])
        ).reshape(batch*out_h*out_w, kh*kw*c)
        return cols, out_h, out_w, (pt,pb,pl,pr), x_p.shape

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass: convolution as matrix multiplication."""
        x = cuda.asarray(x)
        self.last_x = x
        
        # Convert to column format
        cols, out_h, out_w, pads, padded_shape = self._im2col(x)
        
        # Reshape weights and compute convolution
        W_col = self.params['W'].reshape(-1, self.filters)
        out = cols @ W_col
        
        if self.use_bias:
            out = out + self.params['b']
        
        # Reshape to output dimensions
        batch = x.shape[0]
        out = out.reshape(batch, out_h, out_w, self.filters)
        
        # Cache for backward pass
        self.cache = (cols, W_col, out_h, out_w, pads, padded_shape)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients and perform col2im.
        
        Args:
            grad: Gradient from subsequent layer
            
        Returns:
            Gradient w.r.t. input
        """
        cols, W_col, out_h, out_w, pads, padded_shape = self.cache
        kh, kw = self.kernel_size
        batch: int = self.last_x.shape[0]
        xp = cuda.get_array_module(grad)
        
        grad_2d: np.ndarray = grad.reshape(batch * out_h * out_w, self.filters)
        
        # Gradient w.r.t. weights
        dW_col: np.ndarray = cols.T @ grad_2d
        self.grads['W'][...] = dW_col.reshape(kh, kw, self.last_x.shape[3], self.filters)
        
        if self.use_bias:
            self.grads['b'][...] = xp.sum(grad_2d, axis=0)
        
        # Gradient w.r.t. input: col2im operation
        dcols: np.ndarray = grad_2d @ W_col.T
        
        # Reconstruct gradient in image space
        pt, pb, pl, pr = pads
        _, h_p, w_p, c = padded_shape
        dx_p: np.ndarray = xp.zeros((batch, h_p, w_p, c), dtype=self.last_x.dtype)
        
        # Use numba acceleration if available
        if numba_ops.is_numba_available() and xp is np:
            numba_ops.col2im_backward_numba(
                cuda.to_cpu(dcols), dx_p, batch, out_h, out_w, kh, kw, c, self.stride
            )
        else:
            # Fallback: distribute gradients to input positions
            dcols_r: np.ndarray = dcols.reshape(batch, out_h, out_w, kh, kw, c)
            for i in range(out_h):
                i_pos: int = i * self.stride
                for j in range(out_w):
                    j_pos: int = j * self.stride
                    dx_p[:, i_pos:i_pos+kh, j_pos:j_pos+kw, :] += dcols_r[:, i, j, :, :, :]
        
        # Remove padding
        if self.padding == 'same' and (pt > 0 or pb > 0 or pl > 0 or pr > 0):
            dx: np.ndarray = dx_p[:, pt:dx_p.shape[1]-pb if pb > 0 else None, 
                                    pl:dx_p.shape[2]-pr if pr > 0 else None, :]
        else:
            dx = dx_p
        
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Conv2D', 'config': {'filters': self.filters, 'kernel_size': self.kernel_size, 'stride': self.stride, 'padding': self.padding, 'use_bias': self.use_bias}}

class MaxPool2D(Layer):
    """Max pooling layer for spatial downsampling."""
    
    def __init__(self, pool_size: Tuple[int, int] = (2,2), stride: Optional[int] = None) -> None:
        super().__init__()
        if not isinstance(pool_size, tuple) or len(pool_size) != 2:
            raise ValueError(f"pool_size must be a tuple of 2 integers, got {pool_size}")
        if pool_size[0] <= 0 or pool_size[1] <= 0:
            raise ValueError(f"pool_size dimensions must be positive, got {pool_size}")
        
        self.pool_size: Tuple[int, int] = pool_size
        self.stride: int = stride or pool_size[0]
        
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        
        self.trainable: bool = False
        self.last_x: Optional[np.ndarray] = None
        self.max_idx: Optional[np.ndarray] = None
        self.cache: Optional[Tuple] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        _, h, w, c = input_shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.output_shape = (input_shape[0], out_h, out_w, c)
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = cuda.asarray(x)
        self.last_x = x
        batch, h, w, c = x.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        xp = cuda.get_array_module(x)
        # Fast path when stride == pool size: reshape + max
        if self.stride == ph and self.stride == pw:
            x_reshaped = x[:, :out_h*ph, :out_w*pw, :].reshape(batch, out_h, ph, out_w, pw, c)
            y = xp.max(x_reshaped, axis=(2,4))
            # store mask indices for backward
            max_mask = (x_reshaped == y[:, :, None, :, None, :])
            self.cache = (max_mask, x_reshaped.shape, (out_h, out_w))
            return y
        # Fallback general case - use numba if available and on CPU
        y = xp.zeros((batch, out_h, out_w, c), dtype=x.dtype)
        self.max_idx = xp.zeros_like(y, dtype=xp.int32)
        
        if numba_ops.is_numba_available() and xp is np:
            # Use numba-accelerated version for CPU
            numba_ops.maxpool_forward_numba(
                cuda.to_cpu(x), y, self.max_idx, batch, h, w, out_h, out_w, ph, pw, c, self.stride
            )
        else:
            # Original implementation
            for i in range(out_h):
                for j in range(out_w):
                    patch = x[:, i*self.stride:i*self.stride+ph, j*self.stride:j*self.stride+pw, :]
                    flat = patch.reshape(batch, ph*pw, c)
                    idx = xp.argmax(flat, axis=1)
                    self.max_idx[:, i, j, :] = idx
                    y[:, i, j, :] = flat[xp.arange(batch)[:,None], idx, xp.arange(c)]
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: route gradients to max positions.
        
        Args:
            grad: Gradient from subsequent layer
            
        Returns:
            Gradient w.r.t. input
        """
        x: np.ndarray = self.last_x
        batch, h, w, c = x.shape
        ph, pw = self.pool_size
        out_h, out_w = grad.shape[1], grad.shape[2]
        xp = cuda.get_array_module(x)
        
        # Fast path for stride == pool_size with cached mask
        if self.cache is not None:
            max_mask, reshaped_shape, spatial = self.cache
            out_h, out_w = spatial
            # Expand grad to match mask broadcast pattern
            grad_expanded: np.ndarray = grad[:, :, None, :, None, :]
            # Distribute gradients only to max locations
            dx_reshaped: np.ndarray = (max_mask * grad_expanded).astype(x.dtype)
            # Reshape back to original dimensions
            dx_temp: np.ndarray = dx_reshaped.reshape(reshaped_shape)
            dx: np.ndarray = xp.zeros_like(x)
            dx[:, :out_h*ph, :out_w*pw, :] = dx_temp.reshape(
                batch, out_h, ph, out_w, pw, c
            ).transpose(0, 1, 3, 2, 4, 5).reshape(batch, out_h*ph, out_w*pw, c)
            return dx
        
        # General case using index-based routing
        dx = xp.zeros_like(x)
        
        # Use numba acceleration if available and on CPU
        if numba_ops.is_numba_available() and xp is np:
            numba_ops.maxpool_backward_numba(
                dx, cuda.to_cpu(grad), self.max_idx, batch, out_h, out_w, ph, pw, c, self.stride
            )
        else:
            # Fallback implementation
            for i in range(out_h):
                for j in range(out_w):
                    idx: np.ndarray = self.max_idx[:, i, j, :]
                    for n in range(batch):
                        for ch in range(c):
                            pos: int = int(idx[n, ch])
                            r: int = pos // pw
                            col: int = pos % pw
                            dx[n, i*self.stride + r, j*self.stride + col, ch] += grad[n, i, j, ch]
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'MaxPool2D', 'config': {'pool_size': self.pool_size, 'stride': self.stride}}

class BatchNorm2D(Layer):
    """Batch normalization for 2D convolutional layers."""
    
    def __init__(self, momentum: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.momentum: float = momentum
        self.eps: float = eps
        self.last_x: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None
        self.batch_mean: Optional[np.ndarray] = None
        self.batch_var: Optional[np.ndarray] = None
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # input: (batch, H, W, C)
        c: int = input_shape[-1]
        self.params['gamma'] = cuda.asarray(np.ones((c,), dtype=np.float32))
        self.params['beta'] = cuda.asarray(np.zeros((c,), dtype=np.float32))
        self.grads['gamma'] = cuda.zeros_like(self.params['gamma'])
        self.grads['beta'] = cuda.zeros_like(self.params['beta'])
        self.running_mean = cuda.asarray(np.zeros((c,), dtype=np.float32))
        self.running_var = cuda.asarray(np.ones((c,), dtype=np.float32))
        self.output_shape = input_shape
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Normalize inputs across spatial dimensions and batch."""
        x = cuda.asarray(x)
        self.last_x = x
        xp = cuda.get_array_module(x)
        
        if training:
            mean = xp.mean(x, axis=(0,1,2))
            var = xp.var(x, axis=(0,1,2))
            self.batch_mean = mean
            self.batch_var = var
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize and scale
        self.x_hat = (x - mean) / xp.sqrt(var + self.eps)
        return self.params['gamma'] * self.x_hat + self.params['beta']

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through batch normalization.
        
        Args:
            grad: Gradient from subsequent layer
            
        Returns:
            Gradient w.r.t. input
        """
        gamma: np.ndarray = self.params['gamma']
        x_hat: np.ndarray = self.x_hat
        xp = cuda.get_array_module(grad)
        
        N: int = int(xp.prod(xp.array(grad.shape[0:3])))  # Number of elements in batch
        self.grads['gamma'][...] = (grad * x_hat).sum(axis=(0, 1, 2))
        self.grads['beta'][...] = grad.sum(axis=(0, 1, 2))
        
        dx_hat: np.ndarray = grad * gamma
        var: np.ndarray = self.batch_var + self.eps
        
        # Gradient of variance and mean
        dvar: np.ndarray = (dx_hat * (self.last_x - self.batch_mean) * -0.5 * var**(-1.5)).sum(axis=(0, 1, 2))
        dmean: np.ndarray = (
            (dx_hat * -1 / xp.sqrt(var)).sum(axis=(0, 1, 2)) + 
            dvar * (-2 * (self.last_x - self.batch_mean)).sum(axis=(0, 1, 2)) / N
        )
        
        # Final gradient w.r.t. input
        dx: np.ndarray = (
            dx_hat / xp.sqrt(var) + 
            dvar * 2 * (self.last_x - self.batch_mean) / N + 
            dmean / N
        )
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'BatchNorm2D', 'config': {'momentum': self.momentum, 'eps': self.eps}}

NAME2LAYER: Dict[str, type] = {
    cls.__name__: cls 
    for cls in [Dense, Flatten, Activation, Dropout, Conv2D, MaxPool2D, BatchNorm2D]
}
