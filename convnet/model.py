"""Model class implementing training and prediction with JAX acceleration."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Iterator
from .layers import Layer, NAME2LAYER
from .losses import NAME2LOSS, Loss
from .optim import NAME2OPT, Optimizer
from . import io, utils
from . import jax_backend as backend
from tqdm import tqdm
import os

class Model:
    """Neural network model for building, training, and inference.
    
    A sequential container for layers with training capabilities including
    early stopping, learning rate scheduling, and GPU acceleration.
    """
    
    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        self.layers: List[Layer] = layers or []
        self.built: bool = False
        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        self._pending_weights: Optional[Dict[str, np.ndarray]] = None

    def add(self, layer: Layer) -> None:
        """Add a layer to the model."""
        self.layers.append(layer)
        self.built = False

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model by initializing all layers with proper shapes.
        
        Args:
            input_shape: Shape tuple including batch dimension (can be None)
        """
        shape: Tuple[Optional[int], ...] = input_shape
        for idx, layer in enumerate(self.layers):
            if not layer.built:
                layer.build(shape)
            # Assign pending weights if available (from load())
            if self._pending_weights is not None:
                for name in layer.params.keys():
                    key: str = f"{idx}_{layer.__class__.__name__}_{name}"
                    if (key in self._pending_weights and 
                        self._pending_weights[key].shape == layer.params[name].shape):
                        layer.params[name] = self._pending_weights[key]
            out_shape: Optional[Tuple[Optional[int], ...]] = layer.output_shape
            if isinstance(out_shape, tuple):
                if out_shape[0] is None and shape[0] is not None:
                    shape = (shape[0],) + tuple(out_shape[1:])
                else:
                    shape = out_shape
        self.built = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through all layers.
        
        Args:
            x: Input array
            training: Whether in training mode (affects dropout, batchnorm)
            
        Returns:
            Output array
        """
        x = backend.asarray(x)  # Ensure input is on the right device
        
        # Build the model if not already built
        if not self.built:
            self.build(x.shape)
        
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through all layers.
        
        Args:
            grad: Gradient from loss function
            
        Returns:
            Gradient w.r.t. input (rarely used)
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions on input data.
        
        Args:
            x: Input array
            batch_size: Batch size for inference
            
        Returns:
            Predictions array
        """
        if not self.built:
            self.build(x.shape)
        x = backend.asarray(x)  # Ensure input is on the right device
        xp = backend.get_array_module()
        outs: List[np.ndarray] = []
        for i in range(0, x.shape[0], batch_size):
            outs.append(self.forward(x[i:i+batch_size], training=False))
        return xp.concatenate(outs, axis=0)

    def compile(self, loss: str, optimizer: str, weight_decay: float = 0.0, 
                clip_norm: Optional[float] = None, **opt_kwargs: Any) -> None:
        """Configure the model for training.
        
        Args:
            loss: Loss function name ('categorical_crossentropy' or 'mse')
            optimizer: Optimizer name ('sgd' or 'adam')
            weight_decay: L2 regularization strength (default: 0.0)
            clip_norm: Gradient clipping threshold (default: None)
            **opt_kwargs: Additional optimizer parameters (e.g., lr, momentum)
            
        Raises:
            ValueError: If loss or optimizer name is invalid
        """
        if loss not in NAME2LOSS:
            raise ValueError(f"Unknown loss '{loss}'. Valid options: {list(NAME2LOSS.keys())}")
        if optimizer not in NAME2OPT:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Valid options: {list(NAME2OPT.keys())}")
        
        self.loss = NAME2LOSS[loss]()
        self.optimizer = NAME2OPT[optimizer](**opt_kwargs)
        
        # Configure regularization
        self.optimizer.configure(weight_decay=weight_decay, clip_norm=clip_norm)

    def fit(self, dataset: Any, epochs: int = 1, batch_size: int = 32, 
            val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
            num_threads: Optional[int] = None, one_hot_labels: bool = True, 
            num_classes: Optional[int] = None, early_stopping: bool = True, 
            patience: int = 10, min_delta: float = 0.0, 
            lr_schedule: Optional[str] = 'plateau', lr_factor: float = 0.5, 
            lr_patience: int = 5, lr_min: float = 1e-6, 
            verbose: bool = True) -> Dict[str, List[Any]]:
        """Train the model on a dataset.
        
        Args:
            dataset: Dataset object with .images and .batches() method
            epochs: Number of training epochs
            batch_size: Batch size for training
            val_data: Optional (X_val, y_val) tuple for validation
            num_threads: Number of threads for data loading
            one_hot_labels: Whether to convert labels to one-hot encoding
            num_classes: Number of classes (required if one_hot_labels=True)
            early_stopping: Whether to use early stopping
            patience: Epochs to wait before early stopping
            min_delta: Minimum improvement to reset patience
            lr_schedule: Learning rate schedule ('plateau' or None)
            lr_factor: Factor to multiply LR by when reducing
            lr_patience: Epochs to wait before reducing LR
            lr_min: Minimum learning rate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (loss, acc, val_acc, lr)
        """
        # Build model if not already built
        if not self.built:
            inferred_shape: Tuple[Optional[int], ...] = (None,) + dataset.images.shape[1:]
            self.build(inferred_shape)
        
        best_metric: float = -np.inf
        epochs_no_improve: int = 0
        lr_wait: int = 0
        history: Dict[str, List[Any]] = {'loss': [], 'acc': [], 'val_acc': [], 'lr': []}
        
        for epoch in range(epochs):
            pbar = tqdm(
                dataset.batches(batch_size, shuffle=True, num_threads=num_threads),
                total=(len(dataset) + batch_size - 1) // batch_size,
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            losses: List[float] = []
            accs: List[float] = []
            
            for X, y in pbar:
                # Convert labels to one-hot if needed
                if one_hot_labels and y.ndim == 1 and num_classes is not None:
                    y_true: np.ndarray = utils.one_hot(y, num_classes)
                else:
                    y_true = y
                
                # Forward pass
                logits: np.ndarray = self.forward(X, training=True)
                loss_val: float = self.loss.forward(logits, y_true)
                
                # Backward pass
                grad: np.ndarray = self.loss.backward()
                self.backward(grad)
                self.optimizer.step(self._params_and_grads())
                
                # Track metrics
                losses.append(float(backend.to_numpy(loss_val)))
                if y.ndim == 1:
                    xp = backend.get_array_module()
                    preds: np.ndarray = xp.argmax(logits, axis=1)
                    acc: float = float(backend.to_numpy(xp.mean(preds == y)))  
                    accs.append(acc)
                
                pbar.set_postfix(loss=np.mean(losses), acc=np.mean(accs) if accs else 0.0)

            # Validation
            val_acc: Optional[float] = None
            if val_data is not None:
                Xv, yv = val_data
                preds_val: np.ndarray = self.predict(Xv)
                if yv.ndim == 1:
                    xp = backend.get_array_module()
                    acc_tensor = xp.mean(xp.argmax(preds_val, axis=1) == yv)
                    val_acc = float(backend.to_numpy(acc_tensor))
                    if verbose:
                        print(f"Val acc: {val_acc:.4f}")

            # Update history
            metric: float = val_acc if val_acc is not None else (np.mean(accs) if accs else -np.inf)
            history['loss'].append(np.mean(losses))
            history['acc'].append(np.mean(accs) if accs else 0.0)
            history['val_acc'].append(val_acc)
            history['lr'].append(getattr(self.optimizer, 'lr', None))

            # Check for improvement
            if metric > best_metric + min_delta:
                best_metric = metric
                epochs_no_improve = 0
                lr_wait = 0
            else:
                epochs_no_improve += 1
                lr_wait += 1

            # Learning rate scheduling
            if lr_schedule == 'plateau' and lr_wait >= lr_patience:
                if hasattr(self.optimizer, 'lr') and self.optimizer.lr > lr_min:
                    old_lr: float = self.optimizer.lr
                    self.optimizer.lr = max(lr_min, self.optimizer.lr * lr_factor)
                    lr_wait = 0
                    if verbose:
                        print(f"LR reduced from {old_lr:.6f} to {self.optimizer.lr:.6f}")

            # Early stopping
            if early_stopping and epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best metric={best_metric:.4f}")
                break

        return history

    def _params_and_grads(self) -> Iterator[Tuple[Any, str, np.ndarray, np.ndarray]]:
        """Yield (layer, param_name, parameter, gradient) tuples from all trainable layers."""
        for layer in self.layers:
            for name, p, g in layer.get_params_and_grads():
                yield layer, name, p, g

    def save(self, path: str) -> None:
        """Save model weights and architecture to disk.
        
        Args:
            path: File path (with .hdf5 extension or without)
        """
        # Collect weights from all layers
        weights: Dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            for name, param in layer.params.items():
                key: str = f"{idx}_{layer.__class__.__name__}_{name}"
                weights[key] = backend.to_numpy(param)  # Ensure weights are on CPU
        
        arch: List[Dict[str, Any]] = utils.serialize_layers(self.layers)
        
        # Save weights
        base, ext = os.path.splitext(path)
        if ext.lower() == '.hdf5':
            io.save_weights_hdf5(path, weights)
        else:
            # Add .npz extension if not present
            weight_path: str = path if path.endswith('.npz') else path + '.npz'
            np.savez(weight_path, **weights)
        
        # Save architecture as JSON
        import json
        with open(base + '.json', 'w') as f:
            json.dump(arch, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load a model from disk.
        
        Args:
            path: Path to model file (.hdf5 or .npz)
            
        Returns:
            Model instance with loaded weights
            
        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model architecture is incompatible
        """
        base, ext = os.path.splitext(path)
        
        # Load architecture
        arch_path = base + '.json'
        if not os.path.exists(arch_path):
            raise FileNotFoundError(
                f"Model architecture file not found: {arch_path}\n"
                f"Both .json and weight files are required to load a model."
            )
        
        try:
            import json
            with open(arch_path, 'r') as f:
                arch = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in architecture file {arch_path}: {e}")
        
        try:
            layers = utils.deserialize_layers(arch)
        except Exception as e:
            raise ValueError(f"Failed to deserialize model architecture: {e}")
        
        model = cls(layers)
        
        # Load weights
        weight_path = path if ext.lower() == '.hdf5' else path + '.npz'
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Model weights file not found: {weight_path}\n"
                f"Both .json and weight files are required to load a model."
            )
        
        try:
            if ext.lower() == '.hdf5':
                weights = io.load_weights_hdf5(path)
            else:
                data = np.load(path + '.npz')
                weights = {k: data[k] for k in data.files}
        except Exception as e:
            raise ValueError(f"Failed to load weights from {weight_path}: {e}")
        
        # Store weights to be assigned during build
        model._pending_weights = weights
        return model

    def summary(self) -> None:
        """Print a summary of the model architecture and parameter counts."""
        print("Model summary:")
        total: int = 0
        for layer in self.layers:
            params: int = sum(p.size for p in layer.params.values())
            total += params
            print(f"{layer.__class__.__name__}: params={params}")
        print(f"Total params: {total}")
