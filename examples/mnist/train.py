"""MNIST Training Script

Train a CNN on MNIST handwritten digits (0-9).

Dataset: MNIST (28x28 grayscale images)
Classes: 10 (digits 0-9)
Training samples: 60,000
Test samples: 10,000

Download: http://yann.lecun.com/exdb/mnist/
Place .gz files in the same directory as this script.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from convnet import Model, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, Dataset
from convnet import jax_backend as backend
from convnet.data import load_mnist_gz


def build_model(num_classes: int = 10) -> Model:
    """Build CNN for MNIST classification.
    
    Architecture:
    - Conv2D(8) -> ReLU -> MaxPool(2x2)
    - Conv2D(16) -> ReLU -> MaxPool(2x2)
    - Flatten -> Dense(64) -> ReLU -> Dropout(0.2)
    - Dense(10)
    
    Returns:
        Configured model
    """
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
    """Main training loop."""
    print("=" * 60)
    print("MNIST Digit Recognition Training")
    print("=" * 60)
    print(f"Backend: {backend.get_device_name()}")
    if backend.is_numexpr_available():
        print("Optimizations: numexpr (element-wise ops)")
    print("=" * 60)
    print()
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading MNIST dataset...")
    train_full, test = load_mnist_gz(script_dir)
    print(f"Training samples: {len(train_full)}")
    print(f"Test samples: {len(test)}")
    
    # Split train/validation (90/10)
    split_idx = int(0.9 * len(train_full))
    train = Dataset(train_full.images[:split_idx], train_full.labels[:split_idx])
    X_val = train_full.images[split_idx:].astype(np.float32) / 255.0
    y_val = train_full.labels[split_idx:]
    
    print(f"Training: {len(train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print()
    
    # Build model
    print("Building model...")
    model = build_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        lr=0.003,
        weight_decay=1e-5,
        clip_norm=1.0
    )
    print("Model ready")
    print()
    
    # Train
    print("Starting training...")
    print()
    history = model.fit(
        train,
        epochs=50,
        batch_size=64,  # Optimized for 3MB L3 cache
        num_classes=10,
        num_threads=4,
        val_data=(X_val, y_val),
        prefetch=2,
        cache_batches=True,
        early_stopping=True,
        patience=10,
        lr_schedule='plateau',
        lr_factor=0.3,
        lr_patience=4
    )
    
    # Results
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    model.summary()
    
    best_val = max([v for v in history['val_acc'] if v is not None])
    final_loss = history['loss'][-1]
    print(f"\nBest validation accuracy: {best_val:.4f}")
    print(f"Final training loss: {final_loss:.4f}")
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    test_images = test.images.astype(np.float32) / 255.0
    test_preds = model.predict(test_images, batch_size=128)
    test_acc = float(np.mean(np.argmax(test_preds, axis=1) == test.labels))
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('mnist_model.hdf5')
    print("\nModel saved to mnist_model.hdf5")
    print("=" * 60)


if __name__ == '__main__':
    main()
