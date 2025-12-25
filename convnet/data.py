"""Data loading utilities (IDX gzip like MNIST) and simple batching with threads."""
from __future__ import annotations
import gzip
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Iterator, Optional, Callable
import os
from . import jax_backend as backend

# IDX format parsing (MNIST)

def _read_idx_gz(path: str) -> np.ndarray:
    """Read IDX format gzip file (MNIST format).
    
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
    """Simple dataset container with efficient batching and preprocessing.
    
    Provides threaded data loading for better performance during training.
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

    def batches(self, batch_size: int, shuffle: bool = True, 
                preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None, 
                num_threads: Optional[int] = None, 
                use_cuda: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches of data with optional preprocessing and CUDA transfer.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data before batching
            preprocess: Optional preprocessing function
            num_threads: Number of threads for data loading
            use_cuda: Whether to transfer to GPU if available
            
        Yields:
            Tuples of (batch_images, batch_labels)
        """
        idx: np.ndarray = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idx)
        images: np.ndarray = self.images[idx]
        labels: np.ndarray = self.labels[idx]
        n: int = len(self)
        
        if num_threads is None:
            num_threads = min(8, os.cpu_count() or 2)
        
        def load_batch(start: int) -> Tuple[np.ndarray, np.ndarray]:
            """Load and preprocess a single batch."""
            end: int = min(start + batch_size, n)
            X: np.ndarray = images[start:end].astype(np.float32) / 255.0
            if preprocess is not None:
                X = preprocess(X)
            y: np.ndarray = labels[start:end]
            # Move to GPU if requested and available
            if use_cuda and backend.USE_JAX:
                X = backend.asarray(X)
                y = backend.asarray(y)
            return X, y
        
        # Use thread pool for prefetching batches
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = []
            for start in range(0, n, batch_size):
                futures.append(ex.submit(load_batch, start))
                # Yield when we have enough prefetched batches
                if len(futures) >= num_threads:
                    X, y = futures.pop(0).result()
                    yield X, y
            # Yield remaining batches
            for fut in futures:
                X, y = fut.result()
                yield X, y


def load_mnist_gz(folder: str) -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset from gzipped IDX files.
    
    Args:
        folder: Directory containing MNIST .gz files
        
    Returns:
        Tuple of (train_dataset, test_dataset)
        
    Raises:
        FileNotFoundError: If MNIST files are missing
        ValueError: If files are corrupted or invalid
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Please create the folder and download MNIST dataset."
        )
    
    train_images = _read_idx_gz(os.path.join(folder, 'train-images-idx3-ubyte.gz'))
    train_labels = _read_idx_gz(os.path.join(folder, 'train-labels-idx1-ubyte.gz'))
    test_images = _read_idx_gz(os.path.join(folder, 't10k-images-idx3-ubyte.gz'))
    test_labels = _read_idx_gz(os.path.join(folder, 't10k-labels-idx1-ubyte.gz'))
    
    # Reshape to (N, H, W, C) format
    train_images = train_images[..., None]
    test_images = test_images[..., None]
    
    return Dataset(train_images, train_labels), Dataset(test_images, test_labels)
