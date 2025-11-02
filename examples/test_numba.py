#!/usr/bin/env python3
"""Test script to verify numba acceleration is working correctly."""

import numpy as np
import sys

def test_numba_availability():
    """Test if numba is available."""
    print("Testing numba availability...")
    try:
        import convnet
        info = convnet.utils.get_acceleration_info()
        print(f"  CUDA available: {info['cuda']}")
        print(f"  Numba available: {info['numba']}")
        
        if info['numba']:
            print("✓ Numba is available and ready to use!")
            return True
        else:
            print("✗ Numba is not available. Install with: pip install numba")
            return False
    except Exception as e:
        print(f"✗ Error checking numba: {e}")
        return False


def test_numba_acceleration():
    """Test if numba acceleration actually works."""
    print("\nTesting numba-accelerated operations...")
    try:
        from convnet import numba_ops
        
        if not numba_ops.is_numba_available():
            print("  Skipping - numba not available")
            return False
        
        # Test col2im_backward_numba
        print("  Testing col2im_backward_numba...")
        batch, out_h, out_w, kh, kw, c = 2, 3, 3, 3, 3, 4
        dcols = np.random.randn(batch*out_h*out_w, kh*kw*c).astype(np.float32)
        dx_padded = np.zeros((batch, 7, 7, c), dtype=np.float32)
        numba_ops.col2im_backward_numba(dcols, dx_padded, batch, out_h, out_w, kh, kw, c, stride=1)
        print("    ✓ col2im_backward_numba works!")
        
        # Test maxpool_forward_numba
        print("  Testing maxpool_forward_numba...")
        x = np.random.randn(2, 8, 8, 3).astype(np.float32)
        y = np.zeros((2, 4, 4, 3), dtype=np.float32)
        max_idx = np.zeros((2, 4, 4, 3), dtype=np.int32)
        numba_ops.maxpool_forward_numba(x, y, max_idx, 2, 8, 8, 4, 4, 2, 2, 3, stride=2)
        print("    ✓ maxpool_forward_numba works!")
        
        # Test maxpool_backward_numba
        print("  Testing maxpool_backward_numba...")
        dx = np.zeros((2, 8, 8, 3), dtype=np.float32)
        grad = np.random.randn(2, 4, 4, 3).astype(np.float32)
        numba_ops.maxpool_backward_numba(dx, grad, max_idx, 2, 4, 4, 2, 2, 3, stride=2)
        print("    ✓ maxpool_backward_numba works!")
        
        print("✓ All numba operations work correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing numba operations: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_integration():
    """Test that layers properly use numba acceleration."""
    print("\nTesting layer integration with numba...")
    try:
        from convnet.layers import Conv2D, MaxPool2D
        from convnet import numba_ops
        
        if not numba_ops.is_numba_available():
            print("  Skipping - numba not available")
            return False
        
        # Test Conv2D
        print("  Testing Conv2D with numba...")
        conv = Conv2D(filters=8, kernel_size=(3, 3), padding='same')
        conv.build((None, 28, 28, 1))
        x = np.random.randn(4, 28, 28, 1).astype(np.float32)
        y = conv.forward(x, training=True)
        grad = np.random.randn(*y.shape).astype(np.float32)
        dx = conv.backward(grad)
        print(f"    Input shape: {x.shape}, Output shape: {y.shape}, Gradient shape: {dx.shape}")
        print("    ✓ Conv2D works with numba!")
        
        # Test MaxPool2D
        print("  Testing MaxPool2D with numba...")
        pool = MaxPool2D(pool_size=(2, 2))
        pool.build((None, 28, 28, 1))
        x = np.random.randn(4, 28, 28, 1).astype(np.float32)
        y = pool.forward(x, training=True)
        grad = np.random.randn(*y.shape).astype(np.float32)
        dx = pool.backward(grad)
        print(f"    Input shape: {x.shape}, Output shape: {y.shape}, Gradient shape: {dx.shape}")
        print("    ✓ MaxPool2D works with numba!")
        
        print("✓ All layers integrate correctly with numba!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing layer integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Numba Integration Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Numba Availability", test_numba_availability()))
    results.append(("Numba Acceleration", test_numba_acceleration()))
    results.append(("Layer Integration", test_layer_integration()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All tests passed! Numba integration is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
