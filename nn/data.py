"""Data loading utilities (IDX gzip like MNIST) and simple batching with threads."""
from __future__ import annotations
import gzip
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Iterator, Optional, Callable
import os
from . import cuda

# IDX format parsing (MNIST)

def _read_idx_gz(path: str) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        import struct
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(shape)

class Dataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels
    def __len__(self):
        return self.images.shape[0]

    def batches(self, batch_size: int, shuffle: bool = True, preprocess: Optional[Callable] = None, num_threads: Optional[int] = None, use_cuda: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idx)
        images = self.images[idx]
        labels = self.labels[idx]
        n = len(self)
        if num_threads is None:
            num_threads = min(8, os.cpu_count() or 2)
        # simple prefetch pipeline
        def load_batch(start):
            end = min(start+batch_size, n)
            X = images[start:end].astype(np.float32) / 255.0
            if preprocess:
                X = preprocess(X)
            y = labels[start:end]
            # Move to GPU if requested and available
            if use_cuda and cuda.USE_CUDA:
                X = cuda.asarray(X)
                y = cuda.asarray(y)
            return X, y
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = []
            for start in range(0, n, batch_size):
                futures.append(ex.submit(load_batch, start))
                if len(futures) >= num_threads:
                    X, y = futures.pop(0).result()
                    yield X, y
            # drain
            for fut in futures:
                X, y = fut.result()
                yield X, y


def load_mnist_gz(folder: str) -> Tuple[Dataset, Dataset]:
    train_images = _read_idx_gz(os.path.join(folder, 'train-images-idx3-ubyte.gz'))
    train_labels = _read_idx_gz(os.path.join(folder, 'train-labels-idx1-ubyte.gz'))
    test_images = _read_idx_gz(os.path.join(folder, 't10k-images-idx3-ubyte.gz'))
    test_labels = _read_idx_gz(os.path.join(folder, 't10k-labels-idx1-ubyte.gz'))
    # reshape to (N,H,W,C)
    train_images = train_images[..., None]
    test_images = test_images[..., None]
    return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
