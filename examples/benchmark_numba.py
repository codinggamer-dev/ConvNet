#!/usr/bin/env python3
"""Benchmark to demonstrate numba acceleration performance gains."""

import numpy as np
import time
import sys

def benchmark_conv2d():
    """Benchmark Conv2D with and without numba."""
    print("=" * 60)
    print("Conv2D Backward Pass Benchmark")
    print("=" * 60)
    
    from convnet.layers import Conv2D
    from convnet import numba_ops
    
    # Create a Conv2D layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')
    conv.build((None, 28, 28, 16))
    
    # Prepare test data
    x = np.random.randn(32, 28, 28, 16).astype(np.float32)
    y = conv.forward(x, training=True)
    grad = np.random.randn(*y.shape).astype(np.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Filters: {conv.filters}, Kernel: {conv.kernel_size}")
    print(f"Numba available: {numba_ops.is_numba_available()}")
    print()
    
    # Warmup (JIT compilation)
    if numba_ops.is_numba_available():
        print("Warming up (JIT compilation)...")
        for _ in range(2):
            _ = conv.backward(grad)
        print("Warmup complete!")
        print()
    
    # Benchmark backward pass
    num_iterations = 10
    print(f"Running {num_iterations} iterations...")
    
    start = time.time()
    for _ in range(num_iterations):
        dx = conv.backward(grad)
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    print(f"Average backward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} iterations/sec")
    
    return avg_time


def benchmark_maxpool():
    """Benchmark MaxPool2D with and without numba."""
    print("=" * 60)
    print("MaxPool2D Forward+Backward Benchmark")
    print("=" * 60)
    
    from convnet.layers import MaxPool2D
    from convnet import numba_ops
    
    # Create a MaxPool2D layer
    pool = MaxPool2D(pool_size=(2, 2), stride=2)
    pool.build((None, 28, 28, 32))
    
    # Prepare test data
    x = np.random.randn(32, 28, 28, 32).astype(np.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Pool size: {pool.pool_size}, Stride: {pool.stride}")
    print(f"Numba available: {numba_ops.is_numba_available()}")
    print()
    
    # Warmup
    if numba_ops.is_numba_available():
        print("Warming up (JIT compilation)...")
        y = pool.forward(x, training=True)
        grad = np.random.randn(*y.shape).astype(np.float32)
        _ = pool.backward(grad)
        print("Warmup complete!")
        print()
    
    # Benchmark forward + backward pass
    num_iterations = 10
    print(f"Running {num_iterations} iterations...")
    
    start = time.time()
    for _ in range(num_iterations):
        y = pool.forward(x, training=True)
        grad = np.random.randn(*y.shape).astype(np.float32)
        dx = pool.backward(grad)
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    print(f"Average forward+backward time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} iterations/sec")
    
    return avg_time


def main():
    print("\nNumba Performance Benchmark")
    print("This benchmark measures the performance of operations")
    print("with numba JIT compilation enabled.")
    print()
    
    try:
        conv_time = benchmark_conv2d()
        pool_time = benchmark_maxpool()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Conv2D backward: {conv_time*1000:.2f} ms per iteration")
        print(f"MaxPool2D fwd+bwd: {pool_time*1000:.2f} ms per iteration")
        print()
        print("Note: These benchmarks show performance with numba enabled.")
        print("Numba provides significant speedups for loop-heavy operations")
        print("like col2im in Conv2D backward pass and general MaxPool2D.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
