"""Data loading utilities for IDX format datasets with efficient batching.

Provides functions to load IDX gzip files (MNIST, EMNIST) and a Dataset class
with threaded batch loading for optimal training performance.
"""
from __future__ import annotations
import gzip
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Iterator, Optional, Callable
from queue import Queue
from threading import Thread
from . import jax_backend as backend


def _read_idx_gz(path: str) -> np.ndarray:
    """Read IDX format gzip file.
    
    Supports the IDX file format used by MNIST, EMNIST, and similar datasets.
    
    Args:
        path: Path to .gz file
        
    Returns:
        NumPy array with the data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please download MNIST dataset files to the specified directory."
        )
    
    try:
        with gzip.open(path, 'rb') as f:
            import struct
            header = f.read(4)
            if len(header) < 4:
                raise ValueError(f"Invalid IDX file: {path} (file too short)")
            
            zero, data_type, dims = struct.unpack('>HBB', header)
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            
            expected_size = np.prod(shape)
            if data.size != expected_size:
                raise ValueError(
                    f"Data size mismatch in {path}: expected {expected_size}, got {data.size}"
                )
            
            return data.reshape(shape)
    except gzip.BadGzipFile:
        raise ValueError(f"Invalid gzip file: {path}")
    except Exception as e:
        raise ValueError(f"Error reading {path}: {e}")

class Dataset:
    """Dataset container with efficient batching and preprocessing.
    
    Provides threaded data loading and optional preprocessing for optimal
    training performance with automatic device transfer (CPU/GPU).
    
    Attributes:
        images: Image data as numpy array
        labels: Label data as numpy array
    """
    
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if len(images) != len(labels):
            raise ValueError(
                f"Images and labels must have same length, got {len(images)} and {len(labels)}"
            )
        self.images: np.ndarray = images
        self.labels: np.ndarray = labels
        
    def __len__(self) -> int:
        return self.images.shape[0]

    def batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        num_threads: Optional[int] = None,
        use_cuda: bool = True,
        prefetch: int = 2,
        cache: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches with async prefetching for maximum throughput.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data before batching
            preprocess: Optional preprocessing function applied to images
            num_threads: Number of worker threads (default: min(8, cpu_count))
            use_cuda: Whether to transfer batches to GPU if available
            prefetch: Number of batches to prefetch (default: 2 for double buffering)
            cache: Cache preprocessed batches in memory (uses more RAM but faster)
            
        Yields:
            Tuples of (batch_images, batch_labels) on appropriate device
        """
        idx: np.ndarray = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idx)
        images: np.ndarray = self.images[idx]
        labels: np.ndarray = self.labels[idx]
        n: int = len(self)
        
        if num_threads is None:
            num_threads = min(8, os.cpu_count() or 2)
        
        # Cache for preprocessed batches
        batch_cache = {} if cache else None
        
        def load_batch(start: int) -> Tuple[np.ndarray, np.ndarray]:
            """Load, normalize, and preprocess a single batch."""
            if batch_cache is not None and start in batch_cache:
                return batch_cache[start]
            
            end: int = min(start + batch_size, n)
            X: np.ndarray = images[start:end].astype(np.float32) / 255.0
            if preprocess is not None:
                X = preprocess(X)
            y: np.ndarray = labels[start:end]
            
            if use_cuda and backend.USE_JAX:
                X = backend.asarray(X)
                y = backend.asarray(y)
            
            if batch_cache is not None:
                batch_cache[start] = (X, y)
            return X, y
        
        # Async prefetching with double buffering
        if prefetch > 0:
            queue: Queue = Queue(maxsize=prefetch)
            
            def producer():
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    for start in range(0, n, batch_size):
                        future = executor.submit(load_batch, start)
                        queue.put(future)
                queue.put(None)  # Sentinel
            
            thread = Thread(target=producer, daemon=True)
            thread.start()
            
            while True:
                item = queue.get()
                if item is None:
                    break
                yield item.result()
        else:
            # Sequential loading (no prefetch)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for start in range(0, n, batch_size):
                    futures.append(executor.submit(load_batch, start))
                    
                    if len(futures) >= num_threads:
                        X, y = futures.pop(0).result()
                        yield X, y
                
                for future in futures:
                    X, y = future.result()
                    yield X, y


def load_dataset_gz(
    folder: str,
    train_images_file: str = 'train-images-idx3-ubyte.gz',
    train_labels_file: str = 'train-labels-idx1-ubyte.gz',
    test_images_file: str = 't10k-images-idx3-ubyte.gz',
    test_labels_file: str = 't10k-labels-idx1-ubyte.gz',
    add_channel_dim: bool = True
) -> Tuple[Dataset, Dataset]:
    """Load dataset from gzipped IDX files.
    
    Universal loader for MNIST, EMNIST, and other IDX format datasets.
    
    Args:
        folder: Directory containing dataset .gz files
        train_images_file: Filename of training images .gz file
        train_labels_file: Filename of training labels .gz file
        test_images_file: Filename of test images .gz file
        test_labels_file: Filename of test labels .gz file
        add_channel_dim: Whether to add channel dimension (N, H, W) -> (N, H, W, 1)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
        
    Raises:
        FileNotFoundError: If dataset files are missing
        ValueError: If files are corrupted or invalid
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Please create the folder and download the dataset."
        )
    
    train_images = _read_idx_gz(os.path.join(folder, train_images_file))
    train_labels = _read_idx_gz(os.path.join(folder, train_labels_file))
    test_images = _read_idx_gz(os.path.join(folder, test_images_file))
    test_labels = _read_idx_gz(os.path.join(folder, test_labels_file))
    
    if add_channel_dim:
        train_images = train_images[..., None]
        test_images = test_images[..., None]
    
    return Dataset(train_images, train_labels), Dataset(test_images, test_labels)


def load_mnist_gz(folder: str) -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset from gzipped IDX files.
    
    Note: Consider using load_dataset_gz() for greater flexibility.
    This function maintained for backward compatibility.
    
    Args:
        folder: Directory containing MNIST .gz files
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    return load_dataset_gz(
        folder,
        train_images_file='train-images-idx3-ubyte.gz',
        train_labels_file='train-labels-idx1-ubyte.gz',
        test_images_file='t10k-images-idx3-ubyte.gz',
        test_labels_file='t10k-labels-idx1-ubyte.gz'
    )
