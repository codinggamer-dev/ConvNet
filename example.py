"""Example usage of the nn module: Train a small CNN on MNIST gzip files.
Assumes MNIST .gz files are in ./mnist_dataset.

Auto thread configuration occurs when importing nn (sets BLAS threads to cpu cores).
"""
import os
from nn import Model  # triggers auto thread setup before numpy heavy ops
import numpy as np
from nn.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, BatchNorm2D, Dropout
from nn import data


def build_mnist_cnn(num_classes=10, lr=1e-3, weight_decay=1e-4, clip_norm=5.0):
    """Construct a small CNN with regularization settings baked in."""
    model = Model([
        Conv2D(8, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Conv2D(16, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(64), Activation('relu'), Dropout(0.2),
        Dense(num_classes)
    ])
    # New: weight decay & gradient clipping configured via compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        lr=lr,
        weight_decay=weight_decay,
        clip_norm=clip_norm
    )
    return model


def main():
    # Load full training and test sets
    train_full, test = data.load_mnist_gz('mnist_dataset')
    num_classes = 10

    # Create a validation split from training data (e.g., last 10%)
    split_idx = int(0.9 * len(train_full))
    train = data.Dataset(train_full.images[:split_idx], train_full.labels[:split_idx])
    val_images = train_full.images[split_idx:]
    val_labels = train_full.labels[split_idx:]
    # Normalize validation images (training batches are normalized internally)
    X_val = val_images.astype(np.float32) / 255.0
    y_val = val_labels

    model = build_mnist_cnn(num_classes)

    # Train with early stopping & learning rate reduction on plateau.
    history = model.fit(
        train,
        epochs=450,
        batch_size=64,
        num_classes=num_classes,
        val_data=(X_val, y_val),
        early_stopping=True,
        patience=500,
        lr_schedule='plateau',
        lr_factor=0.5,
        lr_patience=4,
        verbose=True
    )

    model.summary()

    # Report best validation accuracy
    best_val = max([v for v in history['val_acc'] if v is not None])
    print(f"Best validation accuracy: {best_val:.4f}")

    # Save and reload the model
    model.save('weights.hdf5')
    loaded = Model.load('weights.hdf5')
    # Build with sample input to ensure parameter shapes are realized
    sample = train.images[:4].astype(np.float32) / 255.0
    loaded.build(sample.shape)
    preds = loaded.predict(sample)
    print('Reloaded prediction batch shape:', preds.shape)

if __name__ == '__main__':
    main()
