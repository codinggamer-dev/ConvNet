"""Layer definitions for the nn module.
Pure NumPy implementations of common layers.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Helper weight initializer functions

def glorot_uniform(shape, rng: np.random.Generator):
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    fan_out = shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)

class Layer:
    """Abstract layer base class."""
    def __init__(self):
        self.built = False
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.trainable = True

    def build(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params_and_grads(self):
        for k, v in self.params.items():
            if self.trainable:
                yield v, self.grads[k]

    def to_config(self) -> Dict[str, Any]:
        return {'class': self.__class__.__name__, 'config': {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)

class Dense(Layer):
    def __init__(self, units: int, use_bias: bool = True, rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.rng = rng or np.random.default_rng()

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.params['W'] = glorot_uniform((in_features, self.units), self.rng)
        if self.use_bias:
            self.params['b'] = np.zeros((self.units,), dtype=np.float32)
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])
        self.output_shape = (*input_shape[:-1], self.units)
        self.built = True

    def forward(self, x, training=False):
        self.last_x = x
        y = x @ self.params['W']
        if self.use_bias:
            y = y + self.params['b']
        return y

    def backward(self, grad):
        x = self.last_x
        self.grads['W'][...] = x.reshape(-1, x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        if self.use_bias:
            self.grads['b'][...] = grad.sum(axis=tuple(range(len(grad.shape)-1)))
        return grad @ self.params['W'].T

    def to_config(self):
        return {'class': 'Dense', 'config': {'units': self.units, 'use_bias': self.use_bias}}

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def build(self, input_shape):
        # input_shape: (batch, H, W, C) or (batch, features)
        # Output becomes (batch, H*W*C)
        if len(input_shape) <= 2:
            # already flat
            self.output_shape = input_shape
        else:
            flat_dim = 1
            for d in input_shape[1:]:
                flat_dim *= d
            self.output_shape = (input_shape[0], flat_dim)
        self.built = True

    def forward(self, x, training=False):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.orig_shape)

class Activation(Layer):
    def __init__(self, func: str = 'relu'):
        super().__init__()
        self.func = func
        self.trainable = False

    def forward(self, x, training=False):
        self.last_x = x
        if self.func == 'relu':
            return np.maximum(0, x)
        if self.func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.func == 'tanh':
            return np.tanh(x)
        if self.func == 'softmax':
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        raise ValueError(f"Unknown activation {self.func}")

    def backward(self, grad):
        x = self.last_x
        if self.func == 'relu':
            return grad * (x > 0)
        if self.func == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return grad * s * (1 - s)
        if self.func == 'tanh':
            t = np.tanh(x)
            return grad * (1 - t**2)
        if self.func == 'softmax':
            # assume combined with cross-entropy handled at loss; pass-through
            return grad
        return grad

    def to_config(self):
        return {'class': 'Activation', 'config': {'func': self.func}}

class Dropout(Layer):
    def __init__(self, rate: float = 0.5, rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.rate = rate
        self.rng = rng or np.random.default_rng()
        self.trainable = False

    def forward(self, x, training=False):
        if training:
            self.mask = (self.rng.random(x.shape) >= self.rate).astype(x.dtype)
            return x * self.mask / (1 - self.rate)
        return x

    def backward(self, grad):
        return grad * self.mask / (1 - self.rate)

    def to_config(self):
        return {'class': 'Dropout', 'config': {'rate': self.rate}}

class Conv2D(Layer):
    """Optimized 2D convolution layer using im2col + GEMM (matrix multiply).

    This replaces the earlier naive nested-loop implementation. The heavy lifting
    is delegated to NumPy's optimized BLAS which can leverage multiple cores.
    """
    def __init__(self, filters: int, kernel_size: Tuple[int, int] = (3,3), stride: int = 1, padding: str = 'same', use_bias: bool = True, rng: Optional[np.random.Generator] = None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.rng = rng or np.random.default_rng()

    def build(self, input_shape):
        _, h, w, c = input_shape
        kh, kw = self.kernel_size
        self.params['W'] = glorot_uniform((kh, kw, c, self.filters), self.rng)
        if self.use_bias:
            self.params['b'] = np.zeros((self.filters,), dtype=np.float32)
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])
        if self.padding == 'same':
            out_h = int(np.ceil(h / self.stride))
            out_w = int(np.ceil(w / self.stride))
        else:
            out_h = (h - kh) // self.stride + 1
            out_w = (w - kw) // self.stride + 1
        self.output_shape = (None, out_h, out_w, self.filters)
        self.built = True

    def _compute_padding(self, h, w):
        if self.padding == 'same':
            kh, kw = self.kernel_size
            pad_h_total = max((np.ceil(h / self.stride) - 1) * self.stride + kh - h, 0)
            pad_w_total = max((np.ceil(w / self.stride) - 1) * self.stride + kw - w, 0)
            pad_top = int(pad_h_total // 2)
            pad_bottom = int(pad_h_total - pad_top)
            pad_left = int(pad_w_total // 2)
            pad_right = int(pad_w_total - pad_left)
            return pad_top, pad_bottom, pad_left, pad_right
        return 0,0,0,0

    def _im2col(self, x):
        batch, h, w, c = x.shape
        kh, kw = self.kernel_size
        pt, pb, pl, pr = self._compute_padding(h, w)
        x_p = np.pad(x, ((0,0),(pt,pb),(pl,pr),(0,0)), mode='constant')
        h_p, w_p = x_p.shape[1], x_p.shape[2]
        out_h = (h_p - kh)//self.stride + 1
        out_w = (w_p - kw)//self.stride + 1
        # Extract patches
        cols = np.lib.stride_tricks.as_strided(
            x_p,
            shape=(batch, out_h, out_w, kh, kw, c),
            strides=(x_p.strides[0], self.stride*x_p.strides[1], self.stride*x_p.strides[2], x_p.strides[1], x_p.strides[2], x_p.strides[3])
        ).reshape(batch*out_h*out_w, kh*kw*c)
        return cols, out_h, out_w, (pt,pb,pl,pr), x_p.shape

    def forward(self, x, training=False):
        self.last_x = x
        cols, out_h, out_w, pads, padded_shape = self._im2col(x)
        W_col = self.params['W'].reshape(-1, self.filters)  # (kh*kw*c, F)
        out = cols @ W_col  # (N*out_h*out_w, F)
        if self.use_bias:
            out += self.params['b']
        batch = x.shape[0]
        out = out.reshape(batch, out_h, out_w, self.filters)
        self.cache = (cols, W_col, out_h, out_w, pads, padded_shape)
        return out

    def backward(self, grad):
        cols, W_col, out_h, out_w, pads, padded_shape = self.cache
        kh, kw = self.kernel_size
        batch = self.last_x.shape[0]
        grad_2d = grad.reshape(batch*out_h*out_w, self.filters)
        # Grad weights
        dW_col = cols.T @ grad_2d  # (kh*kw*c, F)
        self.grads['W'][...] = dW_col.reshape(kh, kw, self.last_x.shape[3], self.filters)
        if self.use_bias:
            self.grads['b'][...] = grad_2d.sum(axis=0)
        # Grad input
        dcols = grad_2d @ W_col.T  # (N*out_h*out_w, kh*kw*c)
        # col2im
        pt,pb,pl,pr = pads
        _, h_p, w_p, c = padded_shape
        dx_p = np.zeros((batch, h_p, w_p, c), dtype=self.last_x.dtype)
        # Reconstruct using loops over spatial output (still much faster than naive pixel loops)
        dcols_r = dcols.reshape(batch, out_h, out_w, kh, kw, c)
        for i in range(out_h):
            i_pos = i * self.stride
            for j in range(out_w):
                j_pos = j * self.stride
                dx_p[:, i_pos:i_pos+kh, j_pos:j_pos+kw, :] += dcols_r[:, i, j, :, :, :]
        if self.padding == 'same':
            dx = dx_p[:, pt:dx_p.shape[1]-pb, pl:dx_p.shape[2]-pr, :]
        else:
            dx = dx_p
        return dx

    def to_config(self):
        return {'class': 'Conv2D', 'config': {'filters': self.filters, 'kernel_size': self.kernel_size, 'stride': self.stride, 'padding': self.padding, 'use_bias': self.use_bias}}

class MaxPool2D(Layer):
    def __init__(self, pool_size=(2,2), stride=None):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride or pool_size[0]
        self.trainable = False

    def build(self, input_shape):
        # input_shape: (batch, H, W, C)
        _, h, w, c = input_shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.output_shape = (input_shape[0], out_h, out_w, c)
        self.built = True

    def forward(self, x, training=False):
        self.last_x = x
        batch, h, w, c = x.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        # Fast path when stride == pool size: reshape + max
        if self.stride == ph and self.stride == pw:
            x_reshaped = x[:, :out_h*ph, :out_w*pw, :].reshape(batch, out_h, ph, out_w, pw, c)
            y = x_reshaped.max(axis=(2,4))
            # store mask indices for backward
            max_mask = (x_reshaped == y[:, :, None, :, None, :])
            self.cache = (max_mask, x_reshaped.shape, (out_h, out_w))
            return y
        # Fallback general case
        y = np.zeros((batch, out_h, out_w, c), dtype=x.dtype)
        self.max_idx = np.zeros_like(y, dtype=np.int32)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i*self.stride:i*self.stride+ph, j*self.stride:j*self.stride+pw, :]
                flat = patch.reshape(batch, ph*pw, c)
                idx = flat.argmax(axis=1)
                self.max_idx[:, i, j, :] = idx
                y[:, i, j, :] = flat[np.arange(batch)[:,None], idx, np.arange(c)]
        return y

    def backward(self, grad):
        x = self.last_x
        batch, h, w, c = x.shape
        ph, pw = self.pool_size
        out_h, out_w = grad.shape[1], grad.shape[2]
        if hasattr(self, 'cache'):
            max_mask, reshaped_shape, spatial = self.cache
            out_h, out_w = spatial
            # grad shape: (batch, out_h, out_w, c)
            # expand grad to match mask broadcast pattern
            grad_expanded = grad[:, :, None, :, None, :]
            # distribute gradients only to max locations
            dx_reshaped = (max_mask * grad_expanded).astype(x.dtype)
            # reshape back
            dx_temp = dx_reshaped.reshape(reshaped_shape)
            dx = np.zeros_like(x)
            dx[:, :out_h*ph, :out_w*pw, :] = dx_temp.reshape(batch, out_h, ph, out_w, pw, c).transpose(0,1,3,2,4,5).reshape(batch, out_h*ph, out_w*pw, c)
            return dx
        dx = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                idx = self.max_idx[:, i, j, :]
                for n in range(batch):
                    for ch in range(c):
                        pos = idx[n, ch]
                        r = pos // pw
                        col = pos % pw
                        dx[n, i*self.stride + r, j*self.stride + col, ch] += grad[n, i, j, ch]
        return dx

    def to_config(self):
        return {'class': 'MaxPool2D', 'config': {'pool_size': self.pool_size, 'stride': self.stride}}

class BatchNorm2D(Layer):
    def __init__(self, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def build(self, input_shape):
        # input: (batch, H, W, C)
        c = input_shape[-1]
        self.params['gamma'] = np.ones((c,), dtype=np.float32)
        self.params['beta'] = np.zeros((c,), dtype=np.float32)
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        self.running_mean = np.zeros((c,), dtype=np.float32)
        self.running_var = np.ones((c,), dtype=np.float32)
        self.output_shape = input_shape
        self.built = True

    def forward(self, x, training=False):
        self.last_x = x
        if training:
            mean = x.mean(axis=(0,1,2))
            var = x.var(axis=(0,1,2))
            self.batch_mean = mean
            self.batch_var = var
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.params['gamma'] * self.x_hat + self.params['beta']

    def backward(self, grad):
        gamma = self.params['gamma']
        x_hat = self.x_hat
        N = np.prod(grad.shape[0:3])
        self.grads['gamma'][...] = (grad * x_hat).sum(axis=(0,1,2))
        self.grads['beta'][...] = grad.sum(axis=(0,1,2))
        dx_hat = grad * gamma
        var = self.batch_var + self.eps
        dvar = (dx_hat * (self.last_x - self.batch_mean) * -0.5 * var**(-1.5)).sum(axis=(0,1,2))
        dmean = (dx_hat * -1/np.sqrt(var)).sum(axis=(0,1,2)) + dvar * (-2*(self.last_x - self.batch_mean)).sum(axis=(0,1,2))/N
        dx = dx_hat / np.sqrt(var) + dvar * 2*(self.last_x - self.batch_mean)/N + dmean / N
        return dx

    def to_config(self):
        return {'class': 'BatchNorm2D', 'config': {'momentum': self.momentum, 'eps': self.eps}}

NAME2LAYER = {cls.__name__: cls for cls in [Dense, Flatten, Activation, Dropout, Conv2D, MaxPool2D, BatchNorm2D]}
