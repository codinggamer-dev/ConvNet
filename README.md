# nn – Minimal NumPy Convolutional Neural Network Framework

An educational yet practical miniature deep learning framework for experimenting with convolutional neural networks, written **entirely from scratch** with only:

* `numpy` (tensor math / BLAS)
* `tqdm` (progress bars)
* Optional: `h5py` **or** TensorFlow only for `.hdf5` weight persistence (falls back to `.npz` if absent)
* Standard library (`threading`, `concurrent.futures`, `gzip`, `json`, `tkinter`)

Everything else (layers, forward/backward passes, optimizers, training loop, serialization, GUI) is implemented manually for transparency.

---
## Table of Contents
1. Philosophy & Goals
2. Feature Overview
3. Project Structure
4. Installation (Beginner & Advanced)
5. Quick Start (Training + GUI)
6. Detailed API Reference
   * Model
   * Layers
   * Losses
   * Optimizers
   * Data / Datasets
   * I/O & Serialization
7. Training Loop Internals
8. Performance & Threading
9. GUI Digit Predictor (`mnist_gui.py`)
10. Extending the Framework (Custom Layers / Losses / Optimizers)
11. Serialization Format Details
12. Common Pitfalls & Troubleshooting
13. Roadmap / Ideas

---
## 1. Philosophy & Goals
Provide a concise, readable code base that:
* Teaches core mechanics: tensor shapes, forward passes, manual gradients.
* Stays small (< a few thousand lines) but non-trivial (Conv2D with im2col, BatchNorm, Adam, GUI inference).
* Avoids magic: every parameter and gradient is visible.
* Enables quick prototyping on MNIST-like data without external heavy frameworks.

Not a replacement for PyTorch / TensorFlow – rather a stepping stone to understand them.

It was made as a private school project, to better understand the math process behind a ConvNet, for a school work.

---
## 2. Feature Overview
| Category | Included |
|----------|----------|
| Layers | `Conv2D` (im2col optimized), `Dense`, `Activation (relu/sigmoid/tanh/softmax)`, `Flatten`, `MaxPool2D` (vectorized fast path), `Dropout`, `BatchNorm2D` |
| Losses | Categorical Crossentropy, MSE |
| Optimizers | SGD (momentum), Adam |
| Data | Gzip IDX (MNIST) loader, threaded batch prefetch |
| Training | Manual loop with progress bars, accuracy reporting, optional early stopping & LR scheduling |
| Serialization | JSON architecture + HDF5 (if available) or NPZ fallback |
| Utilities | One-hot encoding, (de)serialization helpers |
| GUI | Tkinter digit drawing + prediction (`mnist_gui.py`) |
| Performance | im2col Conv2D + BLAS, auto thread configuration based on CPU cores |

---
## 3. Project Structure
```
nn/                 # Framework package
  __init__.py       # Auto thread config + exports
  layers.py         # Layer classes
  model.py          # Model class (compile/fit/predict/save/load)
  losses.py         # Loss implementations
  optim.py          # Optimizers
  data.py           # Dataset + batch generator (threaded prefetch)
  io.py             # HDF5 / NPZ I/O
  utils.py          # Helpers (one_hot, (de)serialize)
example.py          # Training example on MNIST gzip data
mnist_gui.py        # Tkinter drawing + inference GUI
mnist_dataset/      # MNIST gzip IDX files (place here)
weights.*           # Saved weights + architecture after training
```

---
## 4. Installation / Setup

### Beginner (Linux / macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy tqdm h5py  # h5py optional
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy tqdm h5py
```

### Optional: TensorFlow Instead of h5py
If you already use TensorFlow, you can rely on it providing HDF5 support; otherwise `h5py` is the lightest choice.

### Reproducibility
To pin versions:
```bash
pip freeze > requirements.txt
```

---
## 5. Quick Start
Train the sample CNN on MNIST (ensure gzip files exist in `mnist_dataset/`). The example now demonstrates:
* Train/validation split
* Early stopping (patience & min_delta)
* Learning rate reduction on plateau (factor, patience, min lr)
* Weight decay (L2) & gradient clipping (stability late training)

```bash
source .venv/bin/activate
python example.py
```

This will:
1. Load MNIST (train/test).
2. Build a small ConvNet.
3. Train for several epochs (adjust inside `example.py`).
4. Save weights to `weights.hdf5` (+ `weights.json`). If HDF5 unavailable → `weights.hdf5.npz`.
5. Reload model and run a sample forward pass.

### GUI Digit Predictor

```bash
source .venv/bin/activate
python mnist_gui.py
```
Draw digits (0–9). The model preprocesses (threshold, center, blur) and predicts with top-3 probabilities. Resizing window keeps canvas responsive.

---
## 6. Detailed API Reference

### 6.1 Model
```python
from nn import Model
model = Model(layers_list)      # or create empty and call model.add(...)
model.compile(loss='categorical_crossentropy', optimizer='adam', lr=1e-3, weight_decay=1e-4, clip_norm=5.0)
history = model.fit(dataset, epochs=5, batch_size=64, num_classes=10, val_data=(X_val, y_val), early_stopping=True)
preds = model.predict(X)        # (N, num_classes)
model.save('my.hdf5')
loaded = Model.load('my.hdf5')
loaded.build((1, 28, 28, 1))    # ensure shapes before predict if not trained in-session
```
Key methods:
* `build(input_shape)` – propagate shapes & initialize params.
* `forward(x, training=False)` – internal forward pass.
* `backward(grad)` – manual reverse traversal updating gradients.
* `fit(dataset, ...)` – training loop.
* `predict(x)` – batched forward (no gradient).
* `save(path)` / `load(path)` – serialization (JSON + weights).

### 6.2 Layers
Each layer subclass implements: `build(input_shape)`, `forward(x, training)`, `backward(grad)`, and `to_config()`.

Included:
* `Conv2D(filters, kernel_size=(3,3), stride=1, padding='same', use_bias=True)` – im2col → GEMM.
* `Dense(units, use_bias=True)`
* `Activation(func)` – `relu`, `sigmoid`, `tanh`, `softmax`.
* `Flatten()`
* `MaxPool2D(pool_size=(2,2), stride=None)` – stride==pool fast path.
* `Dropout(rate)` – inverted dropout (scales activations at train-time).
* `BatchNorm2D(momentum=0.9, eps=1e-5)`

### 6.3 Losses
* `categorical_crossentropy` / `cce` – expects logits (softmax applied internally in loss wrapper).
* `mse` – mean squared error.

All losses store intermediate forward pass data to compute backward gradient via `loss.backward()`.

### 6.4 Optimizers & Regularization
Optimizers:
* `sgd(lr=0.01, momentum=0.0)`
* `adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`

Regularization / stability hooks configured via `model.compile(...)` when the optimizer supports `configure`:
* `weight_decay` (float, default 0.0): L2 penalty applied per-parameter: `grad += weight_decay * param`.
* `clip_norm` (float or None): Global gradient norm clipping prior to parameter update.

Learning Rate Scheduling & Early Stopping (in `model.fit`):
* `lr_schedule='plateau'`: Monitors validation accuracy (if provided) else training accuracy; if no improvement for `lr_patience` epochs, multiplies learning rate by `lr_factor` (down to `lr_min`).
* `early_stopping=True`: Stop when metric hasn't improved by at least `min_delta` for `patience` epochs.

History Tracking:
`fit` returns a dict: `{ 'loss': [...], 'acc': [...], 'val_acc': [...], 'lr': [...] }` for downstream plotting / analysis.

### 6.5 Data
`nn.data.load_mnist_gz(path)` → `(train_dataset, test_dataset)`.
`Dataset.batches(batch_size, shuffle=True, preprocess=None, num_threads=None)` yields normalized float batches.

Threaded prefetch uses `ThreadPoolExecutor` with worker count = `min(8, cpu_cores)` by default.

### 6.6 I/O & Serialization
* Architecture -> JSON list of layer configs.
* Weights -> HDF5 dataset per param or NPZ fallback.
* `Model.load()` defers weight assignment until `build()` (ensures shape match).

---
## 7. Training Loop Internals
For each batch:
1. Forward pass through layers (with `training=True`).
2. Loss forward (also converts logits to probabilities for cross-entropy).
3. Backprop: start from `loss.backward()` gradient.
4. Accumulate gradients per parameter tensor.
5. Optimizer updates parameters in-place.
6. Track loss & accuracy (if integer labels detected).

Gradients are stored in each layer’s `grads` dict parallel to `params`.

---
## 8. Performance & Threading
* **Conv2D** uses im2col + matrix multiply (leveraging NumPy’s BLAS for multi-core speed).
* **MaxPool2D** has a vectorized stride==pool fast path.
* Automatic thread configuration: on `import nn`, environment variables (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) are set to the detected CPU core count **if** not already defined. Disable by setting `NN_DISABLE_AUTO_THREADS=1` before import.
* Data loader thread count may be specified via `num_threads`; else uses up to 8.

To verify multi-core usage you can run (Linux): `top` or `htop` while training.

---
## 9. GUI Digit Predictor (`mnist_gui.py`)
Features:
* Responsive canvas (resizes with window).
* Draw in high resolution internal buffer (280×280) → downscaled / centered to 28×28.
* Preprocessing: threshold, centroid centering, light blur normalization.
* Displays top-3 class probabilities.

Run after training (needs `weights.hdf5` & `weights.json`):
```bash
python mnist_gui.py
```

---
## 10. Extending the Framework
### Custom Layer
```python
from nn.layers import Layer
import numpy as np

class Scale(Layer):
    def __init__(self, factor):
        super().__init__(); self.factor=factor
    def build(self, input_shape):
        self.output_shape = input_shape; self.built=True
    def forward(self, x, training=False):
        self.last_x = x; return x * self.factor
    def backward(self, grad):
        return grad * self.factor
    def to_config(self):
        return {'class':'Scale','config':{'factor':self.factor}}
```
Add to `NAME2LAYER` in `layers.py` for serialization.

### Custom Loss
Implement `forward(y_pred, y_true)` and `backward()`.

### Custom Optimizer
Subclass `Optimizer` with a `step(params_and_grads)` method.

---
## 11. Serialization Format Details
* `weights.json` – list of objects: `{ "class": "Dense", "config": { ... } }`.
* Weight dataset keys: `<layer_index>_<LayerClass>_<param_name>` (e.g., `7_Dense_W`).
* During load weights cached -> assigned on `build()` when shapes known.

---
## 12. Common Pitfalls & Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Always predicts one class | Input preprocessing mismatch | Use provided GUI pipeline or normalize [0,1] consistently |
| Shape mismatch in Dense | Missing `Flatten` before `Dense` | Insert `Flatten()` layer |
| Slow Conv2D | Large images, Python loops | Already optimized (im2col); reduce filters or batch size |
| High memory use | Large batch with im2col | Reduce batch size |
| Single-core usage | BLAS compiled single-threaded | Install multi-threaded NumPy or set env vars manually (auto attempt already) |
| Invalid gradients (NaN) | Too high learning rate | Lower `lr` |

Disable auto thread config: `export NN_DISABLE_AUTO_THREADS=1` before running.

---
## 13. Roadmap / Ideas
* Additional layers: AveragePool, GlobalPooling, Residual blocks.
* Additional learning rate schedulers (cosine, warm restarts) & richer callbacks.
* Mixed precision (float16) experimental path.
* More dataset utilities (CIFAR10 parsing).
* Col2im full vectorization & further Conv2D micro-optimizations.
* Gradient checking utility.

---
### Acknowledgements
Inspired by concepts from classic deep learning frameworks; deliberately minimal to promote understanding.

Also it was made as a private school project, to better understand the math process behind a ConvNet, for a school work.

Happy hacking – explore, modify, break, and learn.
