"""Layer definitions for the nn module.
High-performance implementations with JAX JIT compilation and GPU/TPU support.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Iterator
from . import jax_backend as backend

# Helper weight initializer functions

def glorot_uniform(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Glorot/Xavier uniform initialization for stable training.
    
    Args:
        shape: Shape tuple for the weight tensor
        rng: NumPy random generator for reproducibility
        
    Returns:
        Initialized weight array on appropriate device
    """
    fan_in: int = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    fan_out: int = shape[0]
    limit: float = np.sqrt(6.0 / (fan_in + fan_out))
    weights: np.ndarray = rng.uniform(-limit, limit, size=shape).astype(np.float32)
    return backend.asarray(weights)

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

    def get_params_and_grads(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Yield (param_name, parameter, gradient) tuples for optimizer updates.
        
        The param_name is needed to update the params dict after optimization
        since JAX arrays are immutable.
        """
        if self.trainable:
            for k, v in self.params.items():
                yield k, v, self.grads[k]
    
    def update_param(self, name: str, new_value: np.ndarray) -> None:
        """Update a parameter with a new value.
        
        This is needed because JAX arrays are immutable.
        """
        self.params[name] = new_value

    def to_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        return {'class': self.__class__.__name__, 'config': {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Layer':
        """Reconstruct layer from configuration."""
        return cls(**config)

class Dense(Layer):
    """Fully connected (dense) layer with JIT-compiled operations."""
    
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
            self.params['b'] = backend.asarray(np.zeros((self.units,), dtype=np.float32))
        self.grads['W'] = backend.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = backend.zeros_like(self.params['b'])
        self.output_shape = (*input_shape[:-1], self.units)
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = backend.asarray(x)
        self.last_x = x
        b = self.params.get('b') if self.use_bias else None
        return backend.dense_forward(x, self.params['W'], b)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.last_x
        dx, dW, db = backend.dense_backward(grad, x, self.params['W'])
        self.grads['W'] = dW
        if self.use_bias:
            self.grads['b'] = db
        return dx

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
        xp = backend.get_array_module()
        return xp.reshape(x, (x.shape[0], -1))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        xp = backend.get_array_module()
        return xp.reshape(grad, self.orig_shape)

class Activation(Layer):
    """Activation function layer with JIT-compiled operations."""
    
    def __init__(self, func: str = 'relu') -> None:
        super().__init__()
        valid_funcs = ['relu', 'sigmoid', 'tanh', 'softmax']
        if func not in valid_funcs:
            raise ValueError(f"Unknown activation '{func}'. Valid options: {valid_funcs}")
        self.func: str = func
        self.trainable: bool = False
        self.last_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = backend.asarray(x)
        self.last_x = x
        
        if self.func == 'relu':
            return backend.relu_forward(x)
        elif self.func == 'sigmoid':
            return backend.sigmoid_forward(x)
        elif self.func == 'tanh':
            return backend.tanh_forward(x)
        elif self.func == 'softmax':
            return backend.softmax_forward(x)
        else:
            raise ValueError(f"Unknown activation {self.func}")

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.last_x
        
        if self.func == 'relu':
            return backend.relu_backward(grad, x)
        elif self.func == 'sigmoid':
            return backend.sigmoid_backward(grad, x)
        elif self.func == 'tanh':
            return backend.tanh_backward(grad, x)
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
        x = backend.asarray(x)
        if training:
            mask = (self.rng.random(x.shape) >= self.rate).astype(np.float32)
            self.mask = backend.asarray(mask)
            return x * self.mask / (1 - self.rate)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask / (1 - self.rate)

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Dropout', 'config': {'rate': self.rate}}

class Conv2D(Layer):
    """2D convolution layer with JIT-compiled im2col + GEMM operations.

    Uses JAX for automatic JIT compilation and GPU/TPU acceleration.
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
            self.params['b'] = backend.asarray(np.zeros((self.filters,), dtype=np.float32))
        self.grads['W'] = backend.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = backend.zeros_like(self.params['b'])
        
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.stride))
            out_w = int(np.ceil(w / self.stride))
        else:
            out_h = (h - kh) // self.stride + 1
            out_w = (w - kw) // self.stride + 1
        self.output_shape = (None, out_h, out_w, self.filters)
        self.built = True

    def _compute_padding(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Compute padding amounts for 'same' or 'valid' padding mode."""
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
        batch, h, w, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        kh, kw = int(self.kernel_size[0]), int(self.kernel_size[1])
        pt, pb, pl, pr = self._compute_padding(h, w)
        
        # Fast manual padding (2x faster than np.pad for small pads)
        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            x_p = np.zeros((batch, h + pt + pb, w + pl + pr, c), dtype=x.dtype)
            x_p[:, pt:pt+h, pl:pl+w, :] = x
        else:
            x_p = x
        h_p, w_p = int(x_p.shape[1]), int(x_p.shape[2])
        
        # Use efficient im2col (JAX native or NumPy)
        cols = backend.im2col(x_p, kh, kw, self.stride)
        out_h = (h_p - kh) // self.stride + 1
        out_w = (w_p - kw) // self.stride + 1
        
        return cols, out_h, out_w, (pt, pb, pl, pr), x_p.shape

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass: convolution as matrix multiplication."""
        x = backend.asarray(x)
        self.last_x = x
        xp = backend.get_array_module()
        
        # Convert to column format
        cols, out_h, out_w, pads, padded_shape = self._im2col(x)
        
        # Reshape weights and compute convolution
        W_col = xp.reshape(self.params['W'], (-1, self.filters))
        out = cols @ W_col
        
        if self.use_bias:
            out = out + self.params['b']
        
        # Reshape to output dimensions
        batch = x.shape[0]
        out = xp.reshape(out, (batch, out_h, out_w, self.filters))
        
        # Cache for backward pass
        self.cache = (cols, W_col, out_h, out_w, pads, padded_shape)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients and perform col2im."""
        cols, W_col, out_h, out_w, pads, padded_shape = self.cache
        kh, kw = self.kernel_size
        batch: int = self.last_x.shape[0]
        c = self.last_x.shape[3]
        xp = backend.get_array_module()
        
        grad_2d = xp.reshape(grad, (batch * out_h * out_w, self.filters))
        
        # Gradient w.r.t. weights: cols.T @ grad_2d
        # cols: (N, K), grad_2d: (N, F) -> dW_col: (K, F)
        dW_col = cols.T @ grad_2d
        self.grads['W'] = xp.reshape(dW_col, (kh, kw, c, self.filters))
        
        if self.use_bias:
            # Use einsum which is faster than sum for this pattern
            self.grads['b'] = xp.einsum('ij->j', grad_2d)
        
        # Gradient w.r.t. input: col2im operation
        dcols = grad_2d @ W_col.T
        
        # Reconstruct gradient in image space
        pt, pb, pl, pr = pads
        pad = pt  # Assuming symmetric padding
        
        # Use col2im backward (JAX native or NumPy)
        dx = backend.col2im_backward(dcols, self.last_x.shape, kh, kw, self.stride, pad)
        
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'Conv2D', 'config': {'filters': self.filters, 'kernel_size': self.kernel_size, 'stride': self.stride, 'padding': self.padding, 'use_bias': self.use_bias}}

class MaxPool2D(Layer):
    """Max pooling layer with JIT-compiled operations."""
    
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
        self.max_mask: Optional[np.ndarray] = None
        self.cache: Optional[Tuple] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        _, h, w, c = input_shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.output_shape = (input_shape[0], out_h, out_w, c)
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.last_x = x
        ph, pw = int(self.pool_size[0]), int(self.pool_size[1])
        stride = int(self.stride)
        
        # Use efficient maxpool (stays on device for JAX)
        y, cache = backend.maxpool_forward(x, ph, pw, stride)
        self.cache = cache
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: route gradients to max positions."""
        x = self.last_x
        ph, pw = int(self.pool_size[0]), int(self.pool_size[1])
        stride = int(self.stride)
        
        # Use efficient backward (stays on device for JAX)
        dx = backend.maxpool_backward(grad, self.cache, x, ph, pw, stride)
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'MaxPool2D', 'config': {'pool_size': self.pool_size, 'stride': self.stride}}


class BatchNorm2D(Layer):
    """Batch normalization with JIT-compiled operations."""
    
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
        c: int = input_shape[-1]
        self.params['gamma'] = backend.asarray(np.ones((c,), dtype=np.float32))
        self.params['beta'] = backend.asarray(np.zeros((c,), dtype=np.float32))
        self.grads['gamma'] = backend.zeros_like(self.params['gamma'])
        self.grads['beta'] = backend.zeros_like(self.params['beta'])
        self.running_mean = backend.asarray(np.zeros((c,), dtype=np.float32))
        self.running_var = backend.asarray(np.ones((c,), dtype=np.float32))
        self.output_shape = input_shape
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Normalize inputs across spatial dimensions and batch."""
        x = backend.asarray(x)
        self.last_x = x
        
        result = backend.batch_norm_forward(
            x, self.params['gamma'], self.params['beta'],
            self.running_mean, self.running_var,
            self.momentum, self.eps, training
        )
        
        output, self.x_hat, self.batch_mean, self.batch_var, \
            self.running_mean, self.running_var = result
        
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through batch normalization."""
        dx, dgamma, dbeta = backend.batch_norm_backward(
            grad, self.last_x, self.x_hat,
            self.params['gamma'], self.batch_var, self.eps
        )
        
        self.grads['gamma'] = dgamma
        self.grads['beta'] = dbeta
        
        return dx

    def to_config(self) -> Dict[str, Any]:
        return {'class': 'BatchNorm2D', 'config': {'momentum': self.momentum, 'eps': self.eps}}


NAME2LAYER: Dict[str, type] = {
    cls.__name__: cls 
    for cls in [Dense, Flatten, Activation, Dropout, Conv2D, MaxPool2D, BatchNorm2D]
}
