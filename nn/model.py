"""Model class implementing training and prediction."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any
from .layers import Layer, NAME2LAYER
from .losses import NAME2LOSS
from .optim import NAME2OPT
from . import io, utils, cuda
from tqdm import tqdm
import os

class Model:
    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = layers or []
        self.built = False

    def add(self, layer: Layer):
        self.layers.append(layer)
        self.built = False

    def build(self, input_shape):
        # propagate shapes
        shape = input_shape
        for idx, layer in enumerate(self.layers):
            if not layer.built:
                layer.build(shape)
            # assign pending weights if available
            if hasattr(self, '_pending_weights'):
                for name in layer.params.keys():
                    key = f"{idx}_{layer.__class__.__name__}_{name}"
                    if key in self._pending_weights and self._pending_weights[key].shape == layer.params[name].shape:
                        layer.params[name] = self._pending_weights[key]
            out_shape = layer.output_shape
            if isinstance(out_shape, tuple):
                if out_shape[0] is None:
                    shape = (shape[0],) + tuple(out_shape[1:])
                else:
                    shape = out_shape
        self.built = True

    def forward(self, x, training=False):
        x = cuda.asarray(x)  # Ensure input is on the right device
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x, batch_size: int = 32):
        if not self.built:
            self.build(x.shape)
        x = cuda.asarray(x)  # Ensure input is on the right device
        xp = cuda.get_array_module(x)
        outs = []
        for i in range(0, x.shape[0], batch_size):
            outs.append(self.forward(x[i:i+batch_size], training=False))
        return xp.concatenate(outs, axis=0)

    def compile(self, loss: str, optimizer: str, weight_decay: float = 0.0, clip_norm: float | None = None, **opt_kwargs):
        self.loss = NAME2LOSS[loss]()
        self.optimizer = NAME2OPT[optimizer](**opt_kwargs)
        # configure regularization
        if hasattr(self.optimizer, 'configure'):
            self.optimizer.configure(weight_decay=weight_decay, clip_norm=clip_norm)

    def fit(self, dataset, epochs: int = 1, batch_size: int = 32, val_data=None, num_threads: Optional[int] = None, one_hot_labels: bool = True, num_classes: Optional[int] = None,
            early_stopping: bool = True, patience: int = 10, min_delta: float = 0.0, lr_schedule: str | None = 'plateau', lr_factor: float = 0.5, lr_patience: int = 5, lr_min: float = 1e-6, verbose: bool = True):
        # dataset: object with .images for shape inference & batches() for iteration
        if not self.built:
            inferred_shape = (None,) + dataset.images.shape[1:]
            self.build(inferred_shape)
        best_metric = -np.inf
        epochs_no_improve = 0
        lr_wait = 0
        history = {'loss': [], 'acc': [], 'val_acc': [], 'lr': []}
        for epoch in range(epochs):
            pbar = tqdm(
                dataset.batches(batch_size, shuffle=True, num_threads=num_threads),
                total=(len(dataset) + batch_size - 1) // batch_size,
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            losses: List[float] = []
            accs: List[float] = []
            for X, y in pbar:
                if one_hot_labels and (y.ndim == 1) and num_classes is not None:
                    y_true = utils.one_hot(y, num_classes)
                else:
                    y_true = y
                logits = self.forward(X, training=True)
                loss_val = self.loss.forward(logits, y_true)
                grad = self.loss.backward()
                self.backward(grad)
                self.optimizer.step(self._params_and_grads())
                losses.append(float(cuda.to_cpu(loss_val)))  # Ensure loss is CPU scalar
                if y.ndim == 1:
                    xp = cuda.get_array_module(logits)
                    preds = xp.argmax(logits, axis=1)
                    acc = float(cuda.to_cpu(xp.mean(preds == y)))  # Convert to CPU scalar
                    accs.append(acc)
                pbar.set_postfix(loss=np.mean(losses), acc=np.mean(accs) if accs else 0.0)

            val_acc = None
            if val_data:
                Xv, yv = val_data
                preds_val = self.predict(Xv)
                if yv.ndim == 1:
                    xp = cuda.get_array_module(preds_val)
                    acc_tensor = xp.mean(xp.argmax(preds_val, axis=1) == yv)
                    val_acc = float(cuda.to_cpu(acc_tensor))  # Convert to CPU scalar
                    if verbose:
                        print(f"Val acc: {val_acc:.4f}")
                else:
                    val_acc = 0.0

            metric = val_acc if val_acc is not None else (np.mean(accs) if accs else -np.inf)
            history['loss'].append(np.mean(losses))
            history['acc'].append(np.mean(accs) if accs else 0.0)
            history['val_acc'].append(val_acc)
            history['lr'].append(getattr(self.optimizer, 'lr', None))

            if metric is not None and metric > best_metric + min_delta:
                best_metric = metric
                epochs_no_improve = 0
                lr_wait = 0  # reset LR plateau counter on improvement
            else:
                epochs_no_improve += 1
                lr_wait += 1

            if lr_schedule == 'plateau' and lr_wait >= lr_patience:
                if hasattr(self.optimizer, 'lr') and self.optimizer.lr > lr_min:
                    old_lr = self.optimizer.lr
                    self.optimizer.lr = max(lr_min, self.optimizer.lr * lr_factor)
                    lr_wait = 0
                    if verbose:
                        print(f"LR reduced from {old_lr} to {self.optimizer.lr}")

            if early_stopping and epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best metric={best_metric:.4f}")
                break

        return history

    def _params_and_grads(self):
        for layer in self.layers:
            for p, g in layer.get_params_and_grads():
                yield p, g

    def save(self, path: str):
        # save weights + architecture (json-like npz)
        weights = {}
        idx = 0
        for layer in self.layers:
            for name, param in layer.params.items():
                weights[f"{idx}_{layer.__class__.__name__}_{name}"] = param
            idx += 1
        arch = utils.serialize_layers(self.layers)
        # write weights
        base, ext = os.path.splitext(path)
        if ext.lower() == '.hdf5':
            io.save_weights_hdf5(path, weights)
        else:
            np.savez(path + '.npz', **weights)
        # save architecture json
        import json
        with open(base + '.json', 'w') as f:
            json.dump(arch, f)

    @classmethod
    def load(cls, path: str):
        base, ext = os.path.splitext(path)
        import json
        with open(base + '.json', 'r') as f:
            arch = json.load(f)
        layers = utils.deserialize_layers(arch)
        model = cls(layers)
        # read weights into pending storage; assign during build when shapes known
        if ext.lower() == '.hdf5':
            weights = io.load_weights_hdf5(path)
        else:
            data = np.load(path + '.npz')
            weights = {k: data[k] for k in data.files}
        model._pending_weights = weights
        return model

    def summary(self):
        print("Model summary:")
        total = 0
        for layer in self.layers:
            params = sum(p.size for p in layer.params.values())
            total += params
            print(f"{layer.__class__.__name__}: params={params}")
        print(f"Total params: {total}")
