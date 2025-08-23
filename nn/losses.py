"""Loss functions and utilities."""
from __future__ import annotations
import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # y_true: integer labels or one-hot
        self.y_true = y_true
        self.y_pred = y_pred
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # integer labels
            probs = y_pred - y_pred.max(axis=1, keepdims=True)
            probs = np.exp(probs)
            probs /= probs.sum(axis=1, keepdims=True)
            self.probs = probs
            log_likelihood = -np.log(probs[np.arange(len(y_true)), y_true.reshape(-1)])
            return log_likelihood.mean()
        else:
            probs = y_pred - y_pred.max(axis=1, keepdims=True)
            probs = np.exp(probs)
            probs /= probs.sum(axis=1, keepdims=True)
            self.probs = probs
            return -(y_true * np.log(probs + 1e-12)).sum(axis=1).mean()

    def backward(self):
        y_true = self.y_true
        probs = self.probs
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            grad = probs.copy()
            grad[np.arange(len(y_true)), y_true.reshape(-1)] -= 1
            grad /= len(y_true)
            return grad
        else:
            grad = (probs - y_true) / y_true.shape[0]
            return grad

class MSE(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_pred - y_true)**2).mean()
    def backward(self):
        return 2*(self.y_pred - self.y_true)/self.y_true.size

NAME2LOSS = {
    'categorical_crossentropy': CategoricalCrossentropy,
    'cce': CategoricalCrossentropy,
    'mse': MSE
}
