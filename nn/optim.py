"""Optimizers (pure numpy)."""
from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple

class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr
        self.weight_decay = 0.0
        self.clip_norm = None
    def configure(self, weight_decay: float = 0.0, clip_norm: float | None = None):
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        return self
    def _apply_regularization(self, p, g):
        if self.weight_decay > 0:
            g = g + self.weight_decay * p
        if self.clip_norm is not None:
            norm = np.linalg.norm(g)
            if norm > self.clip_norm and norm > 0:
                g = g * (self.clip_norm / norm)
        return g
    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        raise NotImplementedError
    def reset(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.v = {}
    def step(self, params_and_grads):
        for i, (p, g) in enumerate(params_and_grads):
            g = self._apply_regularization(p, g)
            if self.momentum > 0:
                v = self.v.get(i, 0)
                v = self.momentum * v - self.lr * g
                self.v[i] = v
                p += v
            else:
                p -= self.lr * g

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1=beta1; self.beta2=beta2; self.eps=eps
        self.m={}; self.v={}; self.t=0
    def step(self, params_and_grads):
        self.t += 1
        for i,(p,g) in enumerate(params_and_grads):
            g = self._apply_regularization(p, g)
            m = self.m.get(i, np.zeros_like(g)); v = self.v.get(i, np.zeros_like(g))
            m = self.beta1*m + (1-self.beta1)*g
            v = self.beta2*v + (1-self.beta2)*(g*g)
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[i]=m; self.v[i]=v

NAME2OPT = {'sgd': SGD, 'adam': Adam}
