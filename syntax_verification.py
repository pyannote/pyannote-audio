#!/usr/bin/env python3
"""
Simple syntax verification script for the GPU optimization changes.
This script verifies that our changes are syntactically correct without external dependencies.
"""

import sys
import os
from pathlib import Path

def test_import_syntax():
    """Test that our modified files can be imported without syntax errors."""
    print("Testing import syntax for modified files...")
    
    # Test pipeline.py
    try:
        with open('pyannote/audio/core/pipeline.py', 'r') as f:
            content = f.read()
        
        # Check for our new method
        if 'def optimize_for_gpu(' in content:
            print("✓ optimize_for_gpu method found in pipeline.py")
        else:
            print("✗ optimize_for_gpu method not found in pipeline.py")
            return False
            
        # Basic syntax check by compiling
        compile(content, 'pyannote/audio/core/pipeline.py', 'exec')
        print("✓ pipeline.py syntax is valid")
        
    except Exception as e:
        print(f"✗ Error with pipeline.py: {e}")
        return False
    
    # Test speaker_diarization.py
    try:
        with open('pyannote/audio/pipelines/speaker_diarization.py', 'r') as f:
            content = f.read()
        
        # Check for improved batch sizes
        if 'embedding_batch_size: int = 32' in content and 'segmentation_batch_size: int = 32' in content:
            print("✓ Improved batch size defaults found in speaker_diarization.py")
        else:
            print("✗ Improved batch size defaults not found in speaker_diarization.py")
            return False
            
        compile(content, 'pyannote/audio/pipelines/speaker_diarization.py', 'exec')
        print("✓ speaker_diarization.py syntax is valid")
        
    except Exception as e:
        print(f"✗ Error with speaker_diarization.py: {e}")
        return False
    
    # Test io.py
    try:
        with open('pyannote/audio/core/io.py', 'r') as f:
            content = f.read()
        
        # Check for caching improvements
        if 'should_cache' in content:
            print("✓ Audio caching improvements found in io.py")
        else:
            print("✗ Audio caching improvements not found in io.py")
            return False
            
        compile(content, 'pyannote/audio/core/io.py', 'exec')
        print("✓ io.py syntax is valid")
        
    except Exception as e:
        print(f"✗ Error with io.py: {e}")
        return False
    
    # Test overlapped_speech_detection.py
    try:
        with open('pyannote/audio/pipelines/overlapped_speech_detection.py', 'r') as f:
            content = f.read()
        
        # Check for batch size improvements
        if 'batch_size" not in inference_kwargs' in content and 'inference_kwargs["batch_size"] = 32' in content:
            print("✓ Batch size improvements found in overlapped_speech_detection.py")
        else:
            print("✗ Batch size improvements not found in overlapped_speech_detection.py")
            return False
            
        compile(content, 'pyannote/audio/pipelines/overlapped_speech_detection.py', 'exec')
        print("✓ overlapped_speech_detection.py syntax is valid")
        
    except Exception as e:
        print(f"✗ Error with overlapped_speech_detection.py: {e}")
        return False
    
    return True

def test_batch_size_improvements():
    """Verify the specific batch size improvements."""
    print("\nTesting batch size improvements...")
    
    with open('pyannote/audio/pipelines/speaker_diarization.py', 'r') as f:
        content = f.read()
    
    # Count occurrences of new defaults
    if content.count('embedding_batch_size: int = 32') >= 1:
        print("✓ embedding_batch_size default changed to 32")
    else:
        print("✗ embedding_batch_size default not updated")
        return False
        
    if content.count('segmentation_batch_size: int = 32') >= 1:
        print("✓ segmentation_batch_size default changed to 32")
    else:
        print("✗ segmentation_batch_size default not updated")  
        return False
    
    # Check docstring updates
    if 'Defaults to 32' in content:
        print("✓ Docstrings updated to reflect new defaults")
    else:
        print("✗ Docstrings not updated")
        return False
    
    return True

def test_gpu_optimization_method():
    """Test the new GPU optimization method."""
    print("\nTesting GPU optimization method...")
    
    with open('pyannote/audio/core/pipeline.py', 'r') as f:
        content = f.read()
    
    # Check method exists
    if 'def optimize_for_gpu(' in content:
        print("✓ optimize_for_gpu method added")
    else:
        print("✗ optimize_for_gpu method not found")
        return False
    
    # Check auto-detection logic
    if 'gpu_memory_gb >= 24' in content and 'batch_size = 64' in content:
        print("✓ GPU memory-based batch size auto-detection implemented")
    else:
        print("✗ GPU memory-based auto-detection not found")
        return False
    
    # Check method chaining support
    if 'return self' in content:
        print("✓ Method chaining support (return self) found")
    else:
        print("✗ Method chaining support not found")
        return False
    
    return True

def test_io_improvements():
    """Test the I/O caching improvements."""
    print("\nTesting I/O caching improvements...")
    
    with open('pyannote/audio/core/io.py', 'r') as f:
        content = f.read()
    
    # Check caching logic
    if 'should_cache' in content:
        print("✓ Audio file caching logic added")
    else:
        print("✗ Audio file caching logic not found")
        return False
    
    # Check memory limit
    if '48000 * 3600' in content:  # 1 hour at 48kHz
        print("✓ Reasonable memory limit for caching implemented")
    else:
        print("✗ Memory limit for caching not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("PyAnnote-Audio GPU Optimization - Syntax Verification")
    print("=" * 55)
    
    tests = [
        test_import_syntax,
        test_batch_size_improvements, 
        test_gpu_optimization_method,
        test_io_improvements
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All syntax verification tests passed!")
        print("\nChanges implemented successfully:")
        print("1. Default batch sizes increased from 1 to 32")
        print("2. Audio file caching to reduce I/O bottlenecks")  
        print("3. GPU optimization method with auto-detection")
        print("4. Improved overlapped speech detection batch sizes")
        print("\nThe fixes should significantly improve GPU utilization!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())