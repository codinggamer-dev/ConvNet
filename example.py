"""Example usage of the nn module: Train a small CNN on MNIST gzip files.
Assumes MNIST .gz files are in ./mnist_dataset.

Auto thread configuration occurs when importing nn (sets BLAS threads to cpu cores).
"""
import os
from nn import Model  # triggers auto thread setup before numpy heavy ops
import numpy as np
from nn.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, BatchNorm2D, Dropout
from nn import data


def build_mnist_cnn(num_classes=10):
    model = Model([
        Conv2D(8, (3,3)), Activation('relu'),
        MaxPool2D((2,2)),
        Conv2D(16, (3,3)), Activation('relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(64), Activation('relu'), Dropout(0.2),
        Dense(num_classes)
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', lr=0.001)
    return model


def main():
    train, test = data.load_mnist_gz('mnist_dataset')
    num_classes = 10
    model = build_mnist_cnn(num_classes)
    model.fit(train, epochs=10, batch_size=64, num_classes=num_classes)
    model.summary()
    # save and load
    model.save('weights.hdf5')
    loaded = Model.load('weights.hdf5')
    # build with sample input to ensure shapes
    sample = train.images[:4].astype(np.float32)/255.0
    loaded.build(sample.shape)
    preds = loaded.predict(sample)
    print('Preds shape', preds.shape)

if __name__ == '__main__':
    main()
