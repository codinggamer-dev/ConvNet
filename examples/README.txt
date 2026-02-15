# Training Examples

Train CNN models on MNIST or EMNIST datasets.

## Quick Start

```bash
# Train on MNIST (digits 0-9)
cd examples
python train.py mnist

# Train on EMNIST (letters A-Z)
python train.py emnist
```

## Dataset Setup

### MNIST
1. Download from: http://yann.lecun.com/exdb/mnist/
2. Place files in `mnist/mnist_dataset/`:
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`

### EMNIST  
1. Download from: https://www.nist.gov/itl/products-and-services/emnist-dataset
2. Place files in `emnist/emnist_dataset/`:
   - `emnist-letters-train-images-idx3-ubyte.gz`
   - `emnist-letters-train-labels-idx1-ubyte.gz`
   - `emnist-letters-test-images-idx3-ubyte.gz`
   - `emnist-letters-test-labels-idx1-ubyte.gz`

## GUI Demos

Interactive drawing interfaces for testing trained models:

```bash
# MNIST digit recognition
cd mnist
python gui.py

# EMNIST letter recognition
cd emnist
python emnist_gui.py
```

## Performance

Training speed: ~15-25 it/s (batch_size=64)
- First epoch may be slower (batch preprocessing)
- Later epochs faster (batch caching enabled)

Expected accuracy:
- MNIST: ~98-99%
- EMNIST: ~85-90%

## Training Parameters

Configured in `train.py`:
- Batch size: 64
- Epochs: 50 (with early stopping)
- Learning rate: 0.003 (Adam optimizer)
- Validation split: 90/10
- Batch prefetching: enabled
- Batch caching: enabled

## Model Architecture

```
Input (28x28x1)
  ↓
Conv2D(8, 3x3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Conv2D(16, 3x3) + ReLU
  ↓
MaxPool2D(2x2)
  ↓
Flatten
  ↓
Dense(64) + ReLU + Dropout(0.2)
  ↓
Dense(num_classes) + Softmax
```

Total parameters: ~25K (MNIST), ~26K (EMNIST)
