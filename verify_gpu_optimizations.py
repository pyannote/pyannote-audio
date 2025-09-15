#!/usr/bin/env python3
"""
Simple verification script to test the GPU optimization improvements.
This script checks that the changes work correctly without requiring actual audio processing.
"""

import sys
import torch
from pathlib import Path

# Add the current directory to Python path to import pyannote
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.audio.core.pipeline import Pipeline
    
    print("✓ Successfully imported pyannote.audio modules")
except ImportError as e:
    print(f"✗ Failed to import pyannote.audio: {e}")
    sys.exit(1)

def test_batch_size_defaults():
    """Test that default batch sizes have been improved."""
    print("\n--- Testing Default Batch Sizes ---")
    
    try:
        # Test that we can create a SpeakerDiarization with new defaults
        # Note: This will try to download models, which might fail without proper setup
        # But we can at least check the constructor parameters
        
        # Check the default values in the constructor signature
        import inspect
        sig = inspect.signature(SpeakerDiarization.__init__)
        
        seg_batch_default = sig.parameters['segmentation_batch_size'].default
        emb_batch_default = sig.parameters['embedding_batch_size'].default
        
        print(f"Default segmentation_batch_size: {seg_batch_default}")
        print(f"Default embedding_batch_size: {emb_batch_default}")
        
        if seg_batch_default == 32 and emb_batch_default == 32:
            print("✓ Default batch sizes correctly updated to 32")
            return True
        else:
            print("✗ Default batch sizes not updated correctly")
            return False
            
    except Exception as e:
        print(f"✗ Error testing batch size defaults: {e}")
        return False

def test_optimize_for_gpu_method():
    """Test that the optimize_for_gpu method exists and works."""
    print("\n--- Testing optimize_for_gpu Method ---")
    
    try:
        # Check if the method exists in the Pipeline class
        if hasattr(Pipeline, 'optimize_for_gpu'):
            print("✓ optimize_for_gpu method exists in Pipeline class")
            
            # Test the method signature
            import inspect
            sig = inspect.signature(Pipeline.optimize_for_gpu)
            print(f"Method signature: {sig}")
            
            # Test that we can call it (on a minimal pipeline instance)
            class TestPipeline(Pipeline):
                def __init__(self):
                    super().__init__()
                    self.segmentation_batch_size = 1
                    self.embedding_batch_size = 1
                
                def default_parameters(self):
                    return {}
                
                def classes(self):
                    return ["TEST"]
                
                def apply(self, file, **kwargs):
                    return None
            
            test_pipeline = TestPipeline()
            
            # Test manual batch size setting
            test_pipeline.optimize_for_gpu(batch_size=64)
            if test_pipeline.segmentation_batch_size == 64 and test_pipeline.embedding_batch_size == 64:
                print("✓ Manual batch size setting works correctly")
            else:
                print("✗ Manual batch size setting failed")
                return False
            
            # Test auto-detection
            test_pipeline.optimize_for_gpu()
            print(f"Auto-detected batch sizes: seg={test_pipeline.segmentation_batch_size}, emb={test_pipeline.embedding_batch_size}")
            
            return True
        else:
            print("✗ optimize_for_gpu method not found in Pipeline class")
            return False
            
    except Exception as e:
        print(f"✗ Error testing optimize_for_gpu method: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and provide recommendations."""
    print("\n--- Testing GPU Availability ---")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        
        print(f"✓ CUDA available with {device_count} device(s)")
        print(f"Current device: {current_device} ({device_name})")
        print(f"Memory: {memory_gb:.1f} GB")
        
        # Provide batch size recommendation
        if memory_gb >= 24:
            recommended_batch = 64
        elif memory_gb >= 12:
            recommended_batch = 32
        elif memory_gb >= 8:
            recommended_batch = 16
        else:
            recommended_batch = 8
            
        print(f"Recommended batch size: {recommended_batch}")
        return True
    else:
        print("✗ CUDA not available - will use CPU")
        return False

def main():
    """Run all verification tests."""
    print("PyAnnote-Audio GPU Optimization Verification")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(test_batch_size_defaults())
    test_results.append(test_optimize_for_gpu_method())
    gpu_available = test_gpu_availability()
    
    # Summary
    print("\n--- Summary ---")
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✓ All core functionality tests passed!")
        if gpu_available:
            print("✓ GPU is available for optimal performance")
            print("\nYou can now use the optimized pipeline with:")
            print("  pipeline.to(torch.device('cuda')).optimize_for_gpu()")
        else:
            print("! GPU not available, but optimizations will still help on CPU")
    else:
        print("✗ Some tests failed - please check the implementation")
        return 1
    
    print("\nNext steps:")
    print("1. Install required dependencies if not already done")
    print("2. Set up HuggingFace token for accessing pretrained models")
    print("3. Test with actual audio files using example_optimized_usage.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())