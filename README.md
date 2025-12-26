# 🧠 ConvNet

[![PyPI](https://img.shields.io/badge/PyPI-convnet-blue.svg)](https://pypi.org/project/convnet/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/Numba-CPU%20JIT-red.svg)](https://numba.pydata.org/)
[![JAX](https://img.shields.io/badge/JAX-GPU%2FTPU-green.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Status](https://img.shields.io/badge/Status-Educational-orange.svg)]()

> **A high-performance, educational CNN framework: Numba for CPU, JAX for GPU/TPU**

This project was created as a school assignment with the goal of understanding deep learning from the ground up. It's designed to be **easy to understand** and **learn from**, implementing a complete CNN framework with **Numba acceleration** for fast CPU training and optional **JAX acceleration** for GPU/TPU. The framework uses simple, readable code while delivering excellent performance.

---

## 🌟 Features

### Core Functionality
- ✅ **Pure Python Core** - Clean, educational code
- 🔥 **Complete CNN Support** - Conv2D, MaxPool2D, Flatten, Dense layers
- 📊 **Modern Training** - Batch normalization, dropout, early stopping
- 🎯 **Smart Optimizers** - SGD with momentum and Adam optimizer
- 📈 **Learning Rate Scheduling** - Plateau-based LR reduction
- 💾 **Model Persistence** - Save/load models in HDF5 or NPZ format
- 🔄 **Data Augmentation Ready** - Thread-pooled data loading

### Performance Options
- 🚀 **Numba CPU Acceleration** - Parallel JIT-compiled operations (15+ it/s)
- ⚡ **JAX GPU/TPU Support** - XLA compilation for maximum throughput
- 📦 **OpenBLAS Integration** - SIMD-optimized matrix operations

### Developer Experience
- 📚 **Clean Code** - Well-documented and easy to follow
- 🎓 **Educational** - Built for learning deep learning fundamentals
- 🔧 **Modular Design** - Easy to extend and customize
- 💻 **Examples Included** - MNIST training example and GUI demo

---

## 🚀 Quick Start

### Installation

**Install from PyPI (Recommended):**

```bash
# Install base package (includes Numba for fast CPU)
pip install convnet

# For GPU/TPU support, add JAX:
pip install convnet[jax]      # CPU JAX
pip install convnet[gpu]      # NVIDIA GPU (CUDA 12)
pip install convnet[tpu]      # Google TPU
```

**Install from Source:**

```bash
# Clone the repository
git clone https://github.com/codinggamer-dev/ConvNet.git
cd ConvNet

# Install in development mode
pip install -e .

# Add JAX for GPU acceleration
pip install jax jaxlib  # For CPU JAX
pip install jax[cuda12]  # For NVIDIA GPU
```

### Performance Expectations

| Backend | MNIST Training Speed | Best For |
|---------|---------------------|----------|
| **Numba (CPU)** | **~16 it/s** (model only) | **Fast CPU training** |
| Pure NumPy | ~6-8 it/s | Fallback, compatibility |
| JAX (CPU) | ~10-12 it/s | Development |
| JAX (GPU) | ~50+ it/s | Production training |

> **Note:** Numba uses all available CPU cores with parallel JIT compilation. Actual training speed is ~10-12 it/s due to loss/optimizer/data loading overhead.
> For maximum performance, ensure you have a modern CPU with AVX2 support.

### Your First Neural Network in 10 Lines

```python
from convnet import Model, Conv2D, Activation, MaxPool2D, Flatten, Dense

# Build a simple CNN
model = Model([
    Conv2D(8, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(10), Activation('softmax')
])

# Configure training
model.compile(loss='categorical_crossentropy', optimizer='adam', lr=0.001)

# Train on your data
history = model.fit(train_dataset, epochs=10, batch_size=32, num_classes=10)
```

---

## 📖 Complete MNIST Example

```python
import numpy as np
from convnet import Model, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Dataset
from convnet.data import load_mnist_gz

# Load MNIST data
train_data, test_data = load_mnist_gz('mnist_dataset')

# Build the model
model = Model([
    Conv2D(8, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Conv2D(16, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(64), Activation('relu'), Dropout(0.2),
    Dense(10)
])

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam', lr=0.001)
history = model.fit(train_data, epochs=50, batch_size=128, num_classes=10)

# Save the model
model.save('my_model.hdf5')
```

---

## 🧩 Architecture Components

### Available Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Conv2D(filters, kernel_size)` | 2D Convolutional layer | `filters`, `kernel_size`, `stride`, `padding` |
| `Dense(units)` | Fully connected layer | `units`, `use_bias` |
| `MaxPool2D(pool_size)` | Max pooling layer | `pool_size`, `stride` |
| `Activation(type)` | Activation function | `'relu'`, `'tanh'`, `'sigmoid'`, `'softmax'` |
| `Flatten()` | Reshape to 1D | None |
| `Dropout(rate)` | Dropout regularization | `rate` (0.0 to 1.0) |
| `BatchNorm2D()` | Batch normalization | `momentum`, `epsilon` |

### Optimizers

- **SGD** - Stochastic Gradient Descent with momentum
  ```python
  model.compile(optimizer='sgd', lr=0.01, momentum=0.9)
  ```

- **Adam** - Adaptive Moment Estimation (recommended)
  ```python
  model.compile(optimizer='adam', lr=0.001, beta1=0.9, beta2=0.999)
  ```

### Loss Functions

- `'categorical_crossentropy'` - For multi-class classification
- `'mse'` - Mean Squared Error for regression

---

## 🎮 Examples & Demos

The `examples/` directory contains several demonstrations:

### 1. MNIST Training (`mnist_train-example.py`)
Complete training pipeline with early stopping, LR scheduling, and model persistence.

```bash
python examples/mnist_train-example.py
```

### 2. Interactive GUI Demo (`mnist_gui.py`)
Draw digits and see real-time predictions! Requires tkinter.

```bash
python examples/mnist_gui.py
```

### 3. GPU Training Test (`test_gpu_training.py`)
Benchmark GPU vs CPU performance.

```bash
python examples/test_gpu_training.py
```

### 4. JAX Benchmark (`benchmark_jax.py`)
Compare JAX JIT performance vs NumPy baseline.

```bash
python examples/benchmark_jax.py
```

---

## ⚙️ Advanced Features

### GPU/TPU Acceleration

ConvNet automatically detects and uses available hardware accelerators via JAX:

```bash
# Install with GPU support
pip install convnet[gpu]

# Or for TPU
pip install convnet[tpu]

# Or install JAX with CUDA manually
pip install jax[cuda12]  # For CUDA 12.x
```

The framework will automatically:
- Detect available GPUs/TPUs
- JIT-compile operations with XLA
- Move tensors to accelerator devices
- Handle device transfers transparently

### Regularization

```python
model.compile(
    optimizer='adam',
    lr=0.001,
    weight_decay=1e-4,  # L2 regularization
    clip_norm=5.0        # Gradient clipping
)
```

### Learning Rate Scheduling

```python
history = model.fit(
    dataset,
    lr_schedule='plateau',  # Reduce LR when validation plateaus
    lr_factor=0.5,         # Multiply LR by 0.5
    lr_patience=5,         # Wait 5 epochs before reducing
    lr_min=1e-6           # Minimum learning rate
)
```

### Early Stopping

```python
history = model.fit(
    dataset,
    val_data=(X_val, y_val),
    early_stopping=True,
    patience=10,      # Stop after 10 epochs without improvement
    min_delta=0.001   # Minimum change to qualify as improvement
)
```

---

## 📊 Model Inspection

```python
# Print model architecture and parameter counts
model.summary()

# Output:
# Model summary:
# Conv2D: params=80
# Activation: params=0
# MaxPool2D: params=0
# Conv2D: params=1168
# Activation: params=0
# MaxPool2D: params=0
# Flatten: params=0
# Dense: params=40064
# Activation: params=0
# Dropout: params=0
# Dense: params=650
# Total params: 41962
```

---

## 🔧 Configuration

### Thread Configuration

The framework automatically configures BLAS threads for optimal CPU performance:

```python
import os
os.environ['NN_DISABLE_AUTO_THREADS'] = '1'  # Disable auto-configuration
import convnet
```

### Custom RNG Seeds

For reproducibility:

```python
import numpy as np
rng = np.random.default_rng(seed=42)

model = Model([
    Conv2D(8, (3, 3), rng=rng),
    Dense(10, rng=rng)
])
```

---

## 📚 Understanding the Code

This project is designed for learning. Here's how to explore:

### Start Here
1. **`convnet/layers.py`** - See how Conv2D, Dense, and other layers work
2. **`convnet/model.py`** - Understand forward/backward propagation
3. **`convnet/optim.py`** - Learn how optimizers update weights
4. **`examples/mnist_train-example.py`** - Complete training example

### Key Concepts Implemented
- 🔄 **Backpropagation** - Full gradient computation chain
- 📉 **Gradient Descent** - SGD and Adam optimization
- 🎲 **Weight Initialization** - Glorot/Xavier uniform
- 🧮 **Convolution Math** - JIT-compiled implementation
- 📊 **Batch Normalization** - Running mean/variance tracking
- 🎯 **Softmax & Cross-Entropy** - Numerically stable implementation

---

## 🎯 Project Goals

This framework was built to:
1. **Understand** deep learning by implementing it from scratch
2. **Learn** how CNNs actually work under the hood
3. **Teach** others the fundamentals of neural networks
4. **Provide** a clean, readable codebase for education

**Not for production use** - Use PyTorch, TensorFlow, or JAX for real applications!

---

## 📦 Project Structure

```
ConvNet/
├── convnet/              # Core framework
│   ├── __init__.py       # Package initialization & auto-config
│   ├── layers.py         # Layer implementations
│   ├── model.py          # Model class with training loop
│   ├── optim.py          # Optimizers (SGD, Adam)
│   ├── losses.py         # Loss functions
│   ├── data.py           # Data loading utilities
│   ├── utils.py          # Helper functions
│   ├── jax_backend.py    # JAX acceleration backend
│   └── io.py             # Model save/load
├── examples/             # Example scripts
│   ├── mnist_train-example.py
│   ├── mnist_gui.py
│   └── mnist_gui.py
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
├── LICENSE.md            # MIT License
└── README.md             # This file
```

---

## 🤝 Contributing

This is an educational project, but contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest improvements
- 📖 Improve documentation
- ✨ Add new features

---

## 📝 Requirements

### Core Dependencies
- **Python** 3.8 or higher
- **JAX** ≥ 0.4.0 (high-performance computing! 🚀)
- **jaxlib** ≥ 0.4.0 (JAX runtime)
- **NumPy** ≥ 1.20.0 (compatibility layer)
- **tqdm** ≥ 4.60.0 (progress bars)
- **h5py** ≥ 3.0.0 (model serialization)

### Optional Dependencies
- **jax[cuda12]** (GPU acceleration)
- **jax[tpu]** (TPU acceleration)
- **tkinter** (for GUI demo, usually included with Python)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

Copyright (c) 2025 Tim Bauer

---

## 🙏 Acknowledgments

- Built as a school project to learn deep learning fundamentals
- Inspired by PyTorch and TensorFlow's clean APIs
- Powered by JAX and XLA for high-performance computation
- MNIST dataset by Yann LeCun and Corinna Cortes - the perfect dataset for learning CNNs

---

## 💬 Questions?

Feel free to open an issue on GitHub if you have questions or run into problems!

---

<div align="center">

**Made with ❤️ for learning and education**

⭐ If this helped you understand CNNs better, consider giving it a star! ⭐

</div>
