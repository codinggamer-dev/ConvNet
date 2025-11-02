# 🧠 ConvNet-NumPy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Powered-green.svg)](https://numpy.org/)
[![CUDA Support](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://cupy.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Status](https://img.shields.io/badge/Status-Educational-orange.svg)]()

> **A clean, educational Convolutional Neural Network framework built from scratch using pure Python and NumPy**

This project was created as a school assignment with the goal of understanding deep learning from the ground up. It's designed to be **easy to understand** and **learn from**, implementing a complete CNN framework using only NumPy for core computations. Additional modules are used only for visualization (tqdm), optimization (Numba JIT), and optional GPU acceleration (CuPy).

---

## 🌟 Features

### Core Functionality
- ✅ **Pure NumPy Core** - All neural network math implemented from scratch
- 🔥 **Complete CNN Support** - Conv2D, MaxPool2D, Flatten, Dense layers
- 📊 **Modern Training** - Batch normalization, dropout, early stopping
- 🎯 **Smart Optimizers** - SGD with momentum and Adam optimizer
- 📈 **Learning Rate Scheduling** - Plateau-based LR reduction
- 💾 **Model Persistence** - Save/load models in HDF5 or NPZ format
- 🔄 **Data Augmentation Ready** - Thread-pooled data loading

### Performance Enhancements
- ⚡ **Numba JIT Compilation** - Automatic acceleration of critical operations
- 🚀 **Optional GPU Support** - CUDA acceleration via CuPy
- 🧵 **Multi-threading** - Auto-configured BLAS threads for CPU optimization
- 📦 **Batch Processing** - Efficient mini-batch training

### Developer Experience
- 📚 **Clean Code** - Well-documented and easy to follow
- 🎓 **Educational** - Built for learning deep learning fundamentals
- 🔧 **Modular Design** - Easy to extend and customize
- 💻 **Examples Included** - MNIST training example and GUI demo

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/codinggamer-dev/ConvNet-NumPy.git
cd ConvNet-NumPy

# Install dependencies
pip install -r requirements.txt

# Optional: Install for GPU acceleration (CUDA 11.x)
pip install cupy-cuda11x

# Optional: Install as package
pip install -e .
```

### Your First Neural Network in 10 Lines

```python
from convnet import Model
from convnet.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense

# Build a simple CNN
model = Model([
    Conv2D(8, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(10)
])

# Configure training
model.compile(loss='categorical_crossentropy', optimizer='adam', lr=0.001)

# Train on your data
history = model.fit(train_dataset, epochs=10, batch_size=32)
```

---

## 📖 Complete MNIST Example

Here's a full example training a CNN on MNIST:

```python
import numpy as np
from convnet import Model, data
from convnet.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout

# Load MNIST data
train_data, test_data = data.load_mnist_gz('mnist_dataset')

# Build the model
model = Model([
    Conv2D(8, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Conv2D(16, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(64), Activation('relu'), Dropout(0.2),
    Dense(10)  # 10 classes for MNIST
])

# Compile with Adam optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    lr=0.001,
    weight_decay=1e-4,
    clip_norm=5.0
)

# Create validation split
split_idx = int(0.9 * len(train_data))
X_val = train_data.images[split_idx:].astype(np.float32) / 255.0
y_val = train_data.labels[split_idx:]
train_subset = data.Dataset(train_data.images[:split_idx], train_data.labels[:split_idx])

# Train with early stopping and LR scheduling
history = model.fit(
    train_subset,
    epochs=100,
    batch_size=256,
    num_classes=10,
    val_data=(X_val, y_val),
    early_stopping=True,
    patience=15,
    lr_schedule='plateau',
    lr_factor=0.5,
    lr_patience=4
)

# Save the model
model.save('my_mnist_model.hdf5')

# Later... load and use
loaded_model = Model.load('my_mnist_model.hdf5')
predictions = loaded_model.predict(test_images)
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

### 4. Numba Benchmark (`benchmark_numba.py`)
Compare Numba JIT vs pure NumPy performance.

```bash
python examples/benchmark_numba.py
```

---

## ⚙️ Advanced Features

### GPU Acceleration

ConvNet-NumPy automatically detects and uses CUDA GPUs when CuPy is installed:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

The framework will automatically:
- Move tensors to GPU
- Use GPU-accelerated operations
- Handle CPU ↔ GPU transfers transparently

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
- 🧮 **Convolution Math** - Pure NumPy implementation
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
ConvNet-NumPy/
├── convnet/              # Core framework
│   ├── __init__.py       # Package initialization & auto-config
│   ├── layers.py         # Layer implementations
│   ├── model.py          # Model class with training loop
│   ├── optim.py          # Optimizers (SGD, Adam)
│   ├── losses.py         # Loss functions
│   ├── data.py           # Data loading utilities
│   ├── utils.py          # Helper functions
│   ├── cuda.py           # GPU acceleration wrapper
│   ├── numba_ops.py      # JIT-compiled operations
│   └── io.py             # Model save/load
├── examples/             # Example scripts
│   ├── mnist_train-example.py
│   ├── mnist_gui.py
│   ├── test_gpu_training.py
│   └── benchmark_numba.py
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
- **NumPy** ≥ 1.20.0 (the star of the show! 🌟)
- **tqdm** ≥ 4.60.0 (progress bars)
- **h5py** ≥ 3.0.0 (model serialization)
- **Numba** ≥ 0.56.0 (JIT compilation)

### Optional Dependencies
- **CuPy** ≥ 10.0.0 (GPU acceleration)
- **tkinter** (for GUI demo, usually included with Python)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

Copyright (c) 2025 Tim Bauer

---

## 🙏 Acknowledgments

- Built as a school project to learn deep learning fundamentals
- Inspired by PyTorch and TensorFlow's clean APIs
- Thanks to the NumPy, Numba, and CuPy teams for amazing tools
- MNIST dataset by Yann LeCun and Corinna Cortes - the perfect dataset for learning CNNs

---

## 💬 Questions?

Feel free to open an issue on GitHub if you have questions or run into problems!

---

<div align="center">

**Made with ❤️ for learning and education**

⭐ If this helped you understand CNNs better, consider giving it a star! ⭐

</div>
