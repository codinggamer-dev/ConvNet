"""Loss functions and utilities."""
from __future__ import annotations
import numpy as np
from . import cuda

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # y_true: integer labels or one-hot
        y_pred = cuda.asarray(y_pred)
        y_true = cuda.asarray(y_true)
        self.y_true = y_true
        self.y_pred = y_pred
        xp = cuda.get_array_module(y_pred)
        
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # integer labels
            probs = y_pred - xp.max(y_pred, axis=1, keepdims=True)
            probs = xp.exp(probs)
            probs /= xp.sum(probs, axis=1, keepdims=True)
            self.probs = probs
            log_likelihood = -xp.log(probs[xp.arange(len(y_true)), y_true.reshape(-1)])
            return float(cuda.to_cpu(xp.mean(log_likelihood)))
        else:
            probs = y_pred - xp.max(y_pred, axis=1, keepdims=True)
            probs = xp.exp(probs)
            probs /= xp.sum(probs, axis=1, keepdims=True)
            self.probs = probs
            return float(cuda.to_cpu(-xp.mean(xp.sum(y_true * xp.log(probs + 1e-12), axis=1))))

    def backward(self):
        y_true = self.y_true
        probs = self.probs
        xp = cuda.get_array_module(probs)
        
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            grad = probs.copy()
            grad[xp.arange(len(y_true)), y_true.reshape(-1)] -= 1
            grad /= len(y_true)
            return grad
        else:
            grad = (probs - y_true) / y_true.shape[0]
            return grad

class MSE(Loss):
    def forward(self, y_pred, y_true):
        y_pred = cuda.asarray(y_pred)
        y_true = cuda.asarray(y_true)
        self.y_pred = y_pred
        self.y_true = y_true
        xp = cuda.get_array_module(y_pred)
        return float(cuda.to_cpu(xp.mean((y_pred - y_true)**2)))
        
    def backward(self):
        xp = cuda.get_array_module(self.y_pred)
        return 2*(self.y_pred - self.y_true)/self.y_true.size

NAME2LOSS = {
    'categorical_crossentropy': CategoricalCrossentropy,
    'cce': CategoricalCrossentropy,
    'mse': MSE
}
