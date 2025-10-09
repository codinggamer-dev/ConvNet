# ConvNet-NumPy 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![NumPy](https://img.shields.io/badge/NumPy-blue.svg)](https://numpy.org/)
[![CUDA Support](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://cupy.dev/)

> A minimal, educational convolutional neural network framework built entirely from scratch using only NumPy. Perfect for understanding the mathematics and mechanics behind deep learning.

## ✨ Features

- 🎯 **Pure NumPy Implementation** - No black boxes, every operation is transparent
- 🚀 **GPU Acceleration** - Optional CUDA support via CuPy with automatic fallback
- 🧱 **Complete CNN Stack** - Conv2D, pooling, batch norm, dropout, and more
- 📊 **Training Utilities** - Progress bars, early stopping, learning rate scheduling
- 🎨 **Interactive GUI** - Draw digits and see real-time predictions
- ⚡ **Optimized Performance** - im2col convolutions, multi-threading, vectorized operations
- 💾 **Model Persistence** - Save/load models with weights and architecture

## 🎯 Why ConvNet-NumPy?

This framework was created as an educational tool to demystify deep learning. Unlike PyTorch or TensorFlow, every operation is implemented from scratch, making it perfect for:

- **Students** learning the fundamentals of neural networks
- **Educators** teaching deep learning concepts
- **Researchers** prototyping custom architectures
- **Anyone** curious about what happens "under the hood"

## 🏗️ Architecture

```
nn/                         # Core framework
├── __init__.py             # Auto-configuration and exports
├── layers.py               # Neural network layers (Conv2D, Dense, etc.)
├── model.py                # Model class with training loop
├── losses.py               # Loss functions
├── optim.py                # Optimizers (SGD, Adam)
├── data.py                 # Data loading and preprocessing
├── io.py                   # Model serialization
├── utils.py                # Utility functions
└── cuda.py                 # GPU acceleration support

mnist_train-example.py      # Complete training example
mnist_gui.py                # Interactive digit recognition GUI
mnist_dataset/              # MNIST data files
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/codinggamer-dev/ConvNet-NumPy.git
cd ConvNet-NumPy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies for CPU
pip install -r requirements-cpu.txt

# OR: Install dependencies for CUDA (replace <cuda_version> with your cuda version: 11, 12 or 13)
pip install -r requirements-cuda<cuda_version>.txt
```

### Train Your First Model

```python
from nn import Model
from nn.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
import nn.data as data

# Load MNIST data
train_data, test_data = data.load_mnist_gz('mnist_dataset')

# Build model
model = Model([
    Conv2D(8, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Conv2D(16, (3, 3)), Activation('relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(64), Activation('relu'),
    Dense(10)  # 10 classes for digits 0-9
])

# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam', lr=1e-3)
history = model.fit(train_data, epochs=10, batch_size=64, num_classes=10)

# Save model
model.save('my_model.hdf5')
```

### Try the Interactive GUI

```bash
python mnist_train-example.py      # Train a model first
python mnist_gui.py                # Launch the digit recognition GUI
```

## 📋 Complete Example

Run the full MNIST training example:

```bash
python mnist_train-example.py
```

This will:
1. 🔧 Configure optimal threading for your CPU
2. 📁 Load MNIST dataset
3. 🏗️ Build and compile a CNN
4. 🎯 Train with validation, early stopping, and learning rate scheduling
5. 💾 Save the trained model
6. 📊 Display training metrics

## 🎮 Interactive Demo

After training, launch the GUI to draw and classify digits:

```bash
python mnist_gui.py
```

Features:
- ✏️ Draw digits with your mouse
- 🔍 Real-time preprocessing and prediction
- 📊 See confidence scores for all 10 digit classes
- 🎨 Responsive canvas that adapts to window size

## 💡 Key Components

### Layers
- **Conv2D** - Optimized convolution using im2col transformation
- **Dense** - Fully connected layers
- **Activation** - ReLU, Sigmoid, Tanh, Softmax
- **MaxPool2D** - Max pooling with vectorized fast path
- **BatchNorm2D** - Batch normalization for training stability
- **Dropout** - Regularization during training
- **Flatten** - Reshape for dense layers

### Training Features
- **Optimizers**: SGD with momentum, Adam with adaptive learning rates
- **Regularization**: Weight decay, gradient clipping, dropout
- **Scheduling**: Learning rate reduction on plateau
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Progress Tracking**: Real-time loss and accuracy monitoring

### Performance Optimizations
- **Multi-threading**: Automatic CPU core detection and optimization
- **Vectorization**: NumPy BLAS for efficient matrix operations
- **GPU Acceleration**: Optional CUDA support via CuPy
- **Memory Efficiency**: Optimized data loading and batch processing

## 🔧 Advanced Configuration

### GPU Setup
```bash
# Check CUDA availability
python -c "import nn; print(f'CUDA: {nn.cuda.CUDA_AVAILABLE}')"

# Control CUDA usage
export NN_DISABLE_CUDA=1  # Force CPU
export NN_FORCE_CUDA=1    # Force GPU (with warnings)
```

### Threading Control
```bash
# Disable automatic thread configuration
export NN_DISABLE_AUTO_THREADS=1

# Manual thread control
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

## 📚 Learning Resources

This framework is perfect for understanding:

1. **Forward Propagation** - How data flows through layers
2. **Backpropagation** - How gradients flow backward
3. **Convolution Mathematics** - im2col transformation and matrix multiplication
4. **Optimization Algorithms** - SGD and Adam from first principles
5. **Regularization Techniques** - Dropout, batch normalization, weight decay
6. **Training Dynamics** - Learning rates, schedules, early stopping

## 🛠️ Extending the Framework

### Custom Layer Example
```python
from nn.layers import Layer
import numpy as np

class ScaleLayer(Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    
    def build(self, input_shape):
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, x, training=False):
        self.last_input = x
        return x * self.factor
    
    def backward(self, grad):
        return grad * self.factor
```

## 📊 Performance Tips

### CPU Optimization
- Use larger batch sizes (64-256) for better vectorization
- Ensure multi-threaded BLAS (automatic configuration included)
- Monitor CPU usage with `htop` during training

### GPU Optimization  
- Use even larger batch sizes (128-512) for GPU efficiency
- First epoch may be slower due to CuPy JIT compilation
- Monitor GPU usage with `nvidia-smi`

## 🤝 Contributing

This project welcomes contributions! Areas for improvement:

- 🧱 Additional layer types (ResNet blocks, attention mechanisms)
- 📊 More datasets beyond MNIST
- ⚡ Further performance optimizations
- 📚 Additional educational examples
- 🧪 Gradient checking utilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

Created as an educational project to understand the mathematical foundations of convolutional neural networks. Inspired by the need for transparent, understandable deep learning implementations.

---

**Happy Learning! 🎓** Explore, modify, break, and rebuild - that's how we truly understand deep learning.

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

### GPU Acceleration (CUDA)
For GPU acceleration, install CuPy (requires NVIDIA GPU with CUDA):

```bash
# For CUDA 11.x (check your CUDA version)
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Alternative: install CuPy from conda-forge
conda install -c conda-forge cupy
```

**Note**: The framework automatically detects CUDA availability and falls back to CPU if:
- CuPy is not installed
- No CUDA-capable GPU is available  
- CUDA drivers are not properly configured

Control CUDA usage with environment variables:
```bash
# Disable CUDA (force CPU usage)
export NN_DISABLE_CUDA=1

# Force CUDA (will warn if unavailable)  
export NN_FORCE_CUDA=1
```

Check CUDA status:
```python
import nn
print(f"CUDA Available: {nn.cuda.CUDA_AVAILABLE}")
print(f"Using CUDA: {nn.cuda.USE_CUDA}")
print(f"Device: {nn.cuda.get_device_name()}")
```

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
python mnist_train-example.py
```

This will:
1. Load MNIST (train/test).
2. Build a small ConvNet.
3. Train for several epochs (adjust inside `mnist_train-example.py`).
4. Save weights to `weights.hdf5` (+ `weights.json`). If HDF5 unavailable → `weights.hdf5.npz`.
5. Reload model and run a sample forward pass.

### GUI Digit Predictor

```bash
source .venv/bin/activate
python mnist_gui.py
```
Draw digits (0–9). The model preprocesses (threshold, center, blur) and predicts with top-3 probabilities. Resizing window keeps canvas responsive.

---

## 📖 Detailed API Reference

### Model Class

```python
from nn import Model

# Create model
model = Model()  # Empty model
model.add(Conv2D(32, (3, 3)))  # Add layers one by one
# OR
model = Model([Conv2D(32, (3, 3)), Activation('relu')])  # All at once

# Configure training
model.compile(
    loss='categorical_crossentropy',  # or 'mse'
    optimizer='adam',                 # or 'sgd'
    lr=1e-3,                         # learning rate
    weight_decay=1e-4,               # L2 regularization
    clip_norm=5.0                    # gradient clipping
)

# Train the model
history = model.fit(
    dataset,                         # training data
    epochs=10,                       # number of epochs
    batch_size=64,                   # batch size
    num_classes=10,                  # for one-hot encoding
    val_data=(X_val, y_val),        # validation data
    early_stopping=True,             # enable early stopping
    patience=5,                      # early stopping patience
    min_delta=1e-4,                  # minimum improvement
    lr_schedule='plateau',           # learning rate scheduling
    lr_patience=3,                   # LR schedule patience
    lr_factor=0.5,                   # LR reduction factor
    lr_min=1e-6                      # minimum learning rate
)

# Make predictions
predictions = model.predict(X_test)

# Save and load
model.save('model.hdf5')
loaded_model = Model.load('model.hdf5')
```

### Available Layers

#### Convolutional Layers
```python
# 2D Convolution with im2col optimization
Conv2D(filters=32, kernel_size=(3, 3), stride=1, padding='same', use_bias=True)

# Max pooling with fast vectorized path
MaxPool2D(pool_size=(2, 2), stride=None)  # stride defaults to pool_size

# Batch normalization for training stability
BatchNorm2D(momentum=0.9, eps=1e-5)
```

#### Core Layers
```python
# Fully connected layer
Dense(units=128, use_bias=True)

# Reshape layer for transitioning from conv to dense
Flatten()

# Activation functions
Activation('relu')     # ReLU activation
Activation('sigmoid')  # Sigmoid activation  
Activation('tanh')     # Tanh activation
Activation('softmax')  # Softmax (for classification output)

# Regularization layer
Dropout(rate=0.2)      # Randomly sets 20% of inputs to 0 during training
```

### Loss Functions

```python
# Available losses
'categorical_crossentropy'  # For multi-class classification
'cce'                      # Shorthand for categorical_crossentropy  
'mse'                      # Mean squared error for regression
```

### Optimizers

```python
# Stochastic Gradient Descent
model.compile(optimizer='sgd', lr=0.01, momentum=0.9)

# Adam optimizer (adaptive learning rates)
model.compile(optimizer='adam', lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
```

### Data Loading

```python
import nn.data as data

# Load MNIST from gzip files
train_dataset, test_dataset = data.load_mnist_gz('mnist_dataset/')

# Create batches with preprocessing
for X_batch, y_batch in train_dataset.batches(
    batch_size=64,
    shuffle=True,
    preprocess=lambda x: x / 255.0,  # normalize to [0,1]
    num_threads=4                    # parallel data loading
):
    # Training loop here
    pass
```

---

## 🔍 Understanding the Internals

### Forward Pass Flow
1. **Input** → Conv2D → **im2col transformation** → **matrix multiplication**
2. **Activation** → ReLU/Sigmoid/etc element-wise operations  
3. **Pooling** → MaxPool2D with vectorized sliding window
4. **Normalization** → BatchNorm2D with running statistics
5. **Regularization** → Dropout masks during training
6. **Output** → Dense layers for final predictions

### Backward Pass (Gradient Computation)
1. **Loss gradient** computed from prediction error
2. **Backpropagate** through each layer in reverse order
3. **Accumulate gradients** for each parameter
4. **Apply regularization** (weight decay, gradient clipping)
5. **Update parameters** using optimizer (SGD/Adam)

### Key Optimizations

#### Convolution Optimization
- **im2col transformation**: Converts convolution to matrix multiplication
- **BLAS acceleration**: Leverages optimized linear algebra libraries
- **Memory layout**: Efficient tensor storage and access patterns

#### Threading and Parallelization
- **Automatic CPU detection**: Sets optimal thread count on import
- **Parallel data loading**: Multi-threaded batch preparation
- **NumPy BLAS**: Multi-core matrix operations

#### GPU Acceleration (Optional)
- **CuPy integration**: Seamless NumPy-to-GPU tensor operations
- **Automatic fallback**: Graceful degradation to CPU if GPU unavailable
- **Memory management**: Efficient GPU memory usage and transfers

---

## 🎛️ Advanced Usage

### Custom Training Loop

```python
import nn
from tqdm import tqdm

# Manual training for full control
model = nn.Model([...])
model.compile(...)

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0
    
    for X_batch, y_batch in train_data.batches(batch_size=64):
        # Forward pass
        predictions = model.forward(X_batch, training=True)
        
        # Compute loss
        loss_val = model.loss.forward(predictions, y_batch)
        
        # Backward pass
        loss_grad = model.loss.backward()
        model.backward(loss_grad)
        
        # Update parameters
        model.optimizer.step(model.get_params_and_grads())
        
        # Track metrics
        epoch_loss += loss_val
        if y_batch.ndim == 1:  # integer labels
            epoch_acc += np.mean(np.argmax(predictions, axis=1) == y_batch)
        num_batches += 1
    
    print(f"Epoch {epoch+1}: Loss={epoch_loss/num_batches:.4f}, Acc={epoch_acc/num_batches:.4f}")
```

### Model Inspection

```python
# Examine model architecture
print(model.summary())

# Access layer parameters
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'params'):
        print(f"Layer {i}: {layer.__class__.__name__}")
        for param_name, param_value in layer.params.items():
            print(f"  {param_name}: {param_value.shape}")

# Monitor gradients during training
def check_gradients():
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'grads'):
            for grad_name, grad_value in layer.grads.items():
                grad_norm = np.linalg.norm(grad_value)
                print(f"Layer {i} {grad_name} gradient norm: {grad_norm:.6f}")
```

### Performance Monitoring

```python
import time

# Timing utilities
def time_forward_pass(model, X, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.predict(X)
        times.append(time.time() - start)
    
    print(f"Average forward pass time: {np.mean(times)*1000:.2f}ms")
    print(f"Throughput: {len(X)/np.mean(times):.0f} samples/sec")

# Memory usage (approximate)
def estimate_memory_usage(model, batch_size, input_shape):
    total_params = 0
    total_memory = 0
    
    for layer in model.layers:
        if hasattr(layer, 'params'):
            layer_params = sum(p.size for p in layer.params.values())
            total_params += layer_params
            total_memory += layer_params * 4  # float32 = 4 bytes
    
    # Add activation memory (forward + backward)
    activation_memory = batch_size * np.prod(input_shape) * 4 * len(model.layers) * 2
    total_memory += activation_memory
    
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory: {total_memory/1024/1024:.1f} MB")
```

---

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Slow Training** | High CPU usage, low throughput | Check thread configuration, use GPU if available |
| **Memory Errors** | OOM during training | Reduce batch size, use gradient checkpointing |
| **Poor Convergence** | Loss not decreasing | Lower learning rate, check data preprocessing |
| **Gradient Explosion** | Loss becomes NaN | Enable gradient clipping, lower learning rate |
| **Overfitting** | Training acc > validation acc | Add dropout, weight decay, or more data |
| **Underfitting** | Both accuracies low | Increase model capacity, train longer |

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check CUDA status
import nn
print(f"CUDA Available: {nn.cuda.CUDA_AVAILABLE}")
print(f"Using CUDA: {nn.cuda.USE_CUDA}")
print(f"Device Info: {nn.cuda.get_device_name()}")

# Validate data pipeline
train_data, test_data = nn.data.load_mnist_gz('mnist_dataset/')
X_batch, y_batch = next(iter(train_data.batches(32)))
print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
print(f"Data ranges: X=[{X_batch.min():.3f}, {X_batch.max():.3f}]")
```

### Environment Variables

```bash
# Threading control
export NN_DISABLE_AUTO_THREADS=1    # Disable automatic thread config
export OMP_NUM_THREADS=8            # OpenMP threads
export OPENBLAS_NUM_THREADS=8       # OpenBLAS threads
export MKL_NUM_THREADS=8            # Intel MKL threads

# CUDA control  
export NN_DISABLE_CUDA=1            # Force CPU usage
export NN_FORCE_CUDA=1              # Force GPU usage (warn if unavailable)

# Debugging
export NN_DEBUG=1                   # Enable debug output
```
---

## 🔧 Extending the Framework

### Custom Layer Example
```python
from nn.layers import Layer
import numpy as np

class ScaleLayer(Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    
    def build(self, input_shape):
        self.output_shape = input_shape
        self.built = True
    
    def forward(self, x, training=False):
        self.last_input = x
        return x * self.factor
    
    def backward(self, grad):
        return grad * self.factor
    
    def to_config(self):
        return {'class': 'ScaleLayer', 'config': {'factor': self.factor}}
```

### Custom Loss Function
```python
from nn.losses import Loss
import numpy as np

class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        self.abs_diff = np.abs(self.diff)
        
        # Huber loss combines MSE and MAE
        quadratic = 0.5 * self.diff**2
        linear = self.delta * (self.abs_diff - 0.5 * self.delta)
        
        loss = np.where(self.abs_diff <= self.delta, quadratic, linear)
        return np.mean(loss)
    
    def backward(self):
        # Gradient is linear for large errors, quadratic for small
        grad = np.where(self.abs_diff <= self.delta, 
                       self.diff, 
                       self.delta * np.sign(self.diff))
        return grad / len(grad)
```

### Custom Optimizer
```python
from nn.optim import Optimizer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, lr=0.001, decay_rate=0.9, eps=1e-8):
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.cache = {}
    
    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            param_id = id(param)
            if param_id not in self.cache:
                self.cache[param_id] = np.zeros_like(param)
            
            # Exponential moving average of squared gradients
            self.cache[param_id] = (self.decay_rate * self.cache[param_id] + 
                                   (1 - self.decay_rate) * grad**2)
            
            # Update parameters
            param -= self.lr * grad / (np.sqrt(self.cache[param_id]) + self.eps)
```

---

## 🎯 Educational Use Cases

### 1. Understanding Backpropagation
```python
# Manually trace gradients through a simple network
model = Model([Dense(10), Activation('relu'), Dense(1)])
model.build((1, 5))

# Forward pass with custom input
x = np.random.randn(1, 5)
output = model.forward(x, training=True)

# Manually compute gradients
grad = np.array([[1.0]])  # Start with unit gradient
for layer in reversed(model.layers):
    print(f"Gradient shape entering {layer.__class__.__name__}: {grad.shape}")
    grad = layer.backward(grad)
    if hasattr(layer, 'grads'):
        print(f"  Parameter gradients: {list(layer.grads.keys())}")
```

### 2. Visualizing Learned Features
```python
# Extract and visualize convolutional filters
conv_layer = model.layers[0]  # Assuming first layer is Conv2D
filters = conv_layer.params['W']  # Shape: (out_channels, in_channels, H, W)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i//4, i%4]
    # Normalize filter for visualization
    filt = filters[i, 0]  # First input channel
    filt = (filt - filt.min()) / (filt.max() - filt.min())
    ax.imshow(filt, cmap='gray')
    ax.set_title(f'Filter {i}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 3. Learning Rate Sensitivity Analysis
```python
# Test different learning rates
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
histories = {}

for lr in learning_rates:
    print(f"Testing learning rate: {lr}")
    model = build_mnist_cnn(lr=lr)
    
    # Train for fewer epochs for quick comparison
    history = model.fit(train_data, epochs=3, batch_size=64, num_classes=10)
    histories[lr] = history
    
    print(f"Final loss: {history['loss'][-1]:.4f}")

# Compare convergence curves
import matplotlib.pyplot as plt
for lr, history in histories.items():
    plt.plot(history['loss'], label=f'LR={lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Rate Comparison')
plt.show()
```

---

## 🏆 Project Ideas

### Beginner Projects
1. **MNIST Variations**: Train on rotated digits, noisy images, or different fonts
2. **Architecture Comparison**: Compare shallow vs deep networks on the same task
3. **Regularization Study**: Compare models with/without dropout and batch normalization
4. **Optimization Comparison**: Compare SGD vs Adam convergence

### Intermediate Projects
1. **Custom Dataset**: Adapt the framework for CIFAR-10 or your own image dataset
2. **Transfer Learning**: Use pre-trained features from a larger model
3. **Data Augmentation**: Implement rotation, flipping, and scaling
4. **Ensemble Methods**: Combine multiple models for better accuracy

### Advanced Projects
1. **Residual Connections**: Implement skip connections for deeper networks
2. **Attention Mechanisms**: Add attention layers for sequence or image tasks
3. **Generative Models**: Build a simple autoencoder or variational autoencoder
4. **Gradient Analysis**: Implement gradient checking and visualization tools

---

## 📚 Further Reading

### Deep Learning Fundamentals
- **"Deep Learning" by Goodfellow, Bengio, and Courville** - Comprehensive theoretical foundation
- **"Neural Networks and Deep Learning" by Michael Nielsen** - Excellent online book with interactive examples
- **CS231n Stanford Course** - Convolutional Neural Networks for Visual Recognition

### Implementation Insights
- **"Implementing Neural Networks from Scratch"** - Various online tutorials
- **PyTorch/TensorFlow Source Code** - See how production frameworks handle similar operations
- **NumPy Documentation** - Deep dive into vectorization and broadcasting

### Research Papers
- **"Gradient-Based Learning Applied to Document Recognition" (LeCun et al.)** - Classic CNN paper
- **"Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy)**
- **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al.)**

---

## 🤝 Contributing

We welcome contributions to improve this educational framework! Here's how you can help:

### Areas for Contribution
- 📚 **Documentation**: Improve explanations, add examples, fix typos
- 🧱 **New Layers**: Implement additional layer types (LSTM, Attention, etc.)
- ⚡ **Optimizations**: Improve performance or memory efficiency
- 🧪 **Testing**: Add unit tests for better reliability
- 📊 **Examples**: Create new educational examples and tutorials
- 🐛 **Bug Fixes**: Report and fix issues

### Development Setup
```bash
git clone https://github.com/codinggamer-dev/ConvNet-NumPy.git
cd ConvNet-NumPy
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements-cpu.txt
pip install pytest  # for testing
```

### Submitting Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Submit a pull request with a clear description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

This framework was created as an educational project to demystify the inner workings of convolutional neural networks. It draws inspiration from:

- **Classic deep learning papers** that established the foundations
- **Modern frameworks** like PyTorch and TensorFlow for API design
- **Educational resources** that make complex topics accessible
- **The open-source community** for tools and libraries that make this possible

Special thanks to the NumPy developers for providing the foundation that makes this educational framework possible.

---

**Happy Learning! 🎓**

*Explore, modify, experiment, and most importantly - understand. The goal isn't just to train models, but to truly comprehend the mathematics and mechanics that make deep learning work.*

---

*📧 Questions? Issues? Ideas? Feel free to open an issue on GitHub or contribute to the project!*

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
* ~~CUDA Support~~ ✅ **COMPLETED** - Optional GPU acceleration via CuPy

---
### Acknowledgements
Inspired by concepts from classic deep learning frameworks; deliberately minimal to promote understanding.

Also it was made as a private school project, to better understand the math process behind a ConvNet, for a school work.

Happy hacking – explore, modify, break, and learn.
