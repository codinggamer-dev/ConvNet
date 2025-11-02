#!/usr/bin/env python3
"""
Test script specifically for GPU environments to verify CUDA training works.
Run this on your NVIDIA system to test the fixes.
"""
import os
import sys
import numpy as np

# Force CUDA if available (for testing)
os.environ['NN_FORCE_CUDA'] = '1'

try:
    import convnet
    print("✅ Framework imported successfully")
    print(f"CUDA Available: {convnet.cuda.CUDA_AVAILABLE}")
    print(f"Using CUDA: {convnet.cuda.USE_CUDA}")
    print(f"Device: {convnet.cuda.get_device_name()}")
except Exception as e:
    print(f"❌ Failed to import framework: {e}")
    sys.exit(1)

# Create a minimal training example
def test_gpu_training():
    print("\n=== Testing GPU Training ===")
    
    # Create very small synthetic dataset
    np.random.seed(42)
    images = np.random.randint(0, 255, (32, 28, 28, 1), dtype=np.uint8)
    labels = np.random.randint(0, 10, (32,), dtype=np.uint8)
    dataset = convnet.data.Dataset(images, labels)
    
    # Create simple model
    from convnet.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
    model = convnet.Model([
        Conv2D(4, (3, 3)), Activation('relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(10)
    ])
    
    model.compile('categorical_crossentropy', 'adam', lr=0.01)
    
    print("Model created, starting training...")
    
    try:
        # Train for just 2 epochs to test
        history = model.fit(
            dataset,
            epochs=2,
            batch_size=8,
            num_classes=10,
            early_stopping=False,  # Disable for this test
            verbose=True
        )
        
        print(f"✅ Training completed successfully!")
        print(f"Final loss: {history['loss'][-1]:.4f}")
        print(f"Final accuracy: {history['acc'][-1]:.4f}")
        
        # Test prediction
        sample = images[:4].astype(np.float32) / 255.0
        if convnet.cuda.USE_CUDA:
            sample = convnet.cuda.asarray(sample)
        
        preds = model.predict(sample)
        print(f"Prediction shape: {preds.shape}")
        print(f"Prediction device type: {type(preds)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_array_operations():
    print("\n=== Testing CUDA Array Operations ===")
    
    try:
        # Test basic array operations
        x = np.random.randn(16, 10).astype(np.float32)
        x_gpu = convnet.cuda.asarray(x)
        
        print(f"Original: {type(x)}")
        print(f"GPU version: {type(x_gpu)}")
        
        # Test one_hot with GPU array
        labels = np.random.randint(0, 5, (16,))
        if convnet.cuda.USE_CUDA:
            labels = convnet.cuda.asarray(labels)
        
        oh = convnet.utils.one_hot(labels, 5)
        print(f"One-hot result type: {type(oh)}")
        print(f"One-hot shape: {oh.shape}")
        
        # Test loss computation
        from convnet.losses import CategoricalCrossentropy
        loss = CategoricalCrossentropy()
        
        pred = np.random.randn(16, 5).astype(np.float32)
        if convnet.cuda.USE_CUDA:
            pred = convnet.cuda.asarray(pred)
        
        loss_val = loss.forward(pred, labels)
        print(f"Loss value: {loss_val}, type: {type(loss_val)}")
        
        print("✅ All CUDA operations working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True
    
    # Test CUDA operations
    if not test_cuda_array_operations():
        success = False
    
    # Test training
    if not test_gpu_training():
        success = False
    
    if success:
        print("\n🎉 All GPU tests passed!")
        if convnet.cuda.USE_CUDA:
            print("🚀 Ready for full GPU training!")
        else:
            print("🖥️  Running on CPU (expected if no GPU available)")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)