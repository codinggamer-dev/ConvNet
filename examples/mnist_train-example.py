"""MNIST Training Example

Train a CNN on MNIST handwritten digits using the convnet framework.
Requires MNIST .gz files in ./mnist_dataset subdirectory.

GPU/TPU acceleration via JAX is used automatically when available.
"""
import os
import numpy as np
from convnet import Model, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Dataset
from convnet import jax_backend as backend
from convnet.data import load_mnist_gz


def build_model(num_classes=10):
    """Build a simple CNN for MNIST classification."""
    return Model([
        Conv2D(8, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Conv2D(16, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(64), Activation('relu'), Dropout(0.2),
        Dense(num_classes)
    ])


def main():
    # Display device info
    print("=== ConvNet MNIST Training ===")
    print(f"Device: {backend.get_device_name()}")
    if backend.is_gpu_available():
        print("🚀 GPU acceleration active!")
    else:
        print("🖥️  Running on CPU")
    print()
    
    # Load MNIST data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_path = os.path.join(script_dir, 'mnist_dataset')
    train_full, test = load_mnist_gz(mnist_path)
    
    # Split into train/val (90/10)
    split_idx = int(0.9 * len(train_full))
    train = Dataset(train_full.images[:split_idx], train_full.labels[:split_idx])
    X_val = train_full.images[split_idx:].astype(np.float32) / 255.0
    y_val = train_full.labels[split_idx:]

    # Build and compile model
    model = build_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        lr=0.003,
        weight_decay=1e-5,
        clip_norm=1.0
    )

    # Train
    history = model.fit(
        train,
        epochs=75,
        batch_size=128,
        num_classes=10,
        val_data=(X_val, y_val),
        early_stopping=True,
        patience=8,
        lr_schedule='plateau',
        lr_factor=0.3,
        lr_patience=3
    )

    # Results
    model.summary()
    best_val = max([v for v in history['val_acc'] if v is not None])
    print(f"\nBest validation accuracy: {best_val:.4f}")

    # Save model
    model.save('mnist_model.hdf5')
    print("Model saved to mnist_model.hdf5")

if __name__ == '__main__':
    main()
