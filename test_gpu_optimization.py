#!/usr/bin/env python3
"""
Test script to verify GPU optimization improvements for pyannote-audio speaker diarization.

This script demonstrates the usage of the new optimization features:
1. Improved default batch sizes (32 instead of 1)
2. Audio file caching to reduce IO bottlenecks
3. New optimize_for_gpu() method for easy configuration
"""

import torch
from pyannote.audio import Pipeline

def test_gpu_optimization():
    """Test the GPU optimization features."""
    
    print("Testing pyannote-audio GPU optimization improvements...")
    
    # Load a pretrained pipeline
    # Note: You'll need to accept user conditions at https://hf.co/pyannote/speaker-diarization-3.1
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="YOUR_TOKEN_HERE"  # Replace with your HuggingFace token
    )
    
    print(f"Default segmentation batch size: {pipeline.segmentation_batch_size}")
    print(f"Default embedding batch size: {pipeline.embedding_batch_size}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipeline.to(device)
        print(f"Pipeline moved to: {device}")
        
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        # Optimize for GPU usage with auto-detection
        pipeline.optimize_for_gpu()
        print(f"After optimization - segmentation batch size: {pipeline.segmentation_batch_size}")
        print(f"After optimization - embedding batch size: {pipeline.embedding_batch_size}")
        
        # You can also manually set batch sizes
        pipeline.optimize_for_gpu(batch_size=64)
        print(f"After manual optimization - segmentation batch size: {pipeline.segmentation_batch_size}")
        print(f"After manual optimization - embedding batch size: {pipeline.embedding_batch_size}")
        
    else:
        print("CUDA not available. Running on CPU.")
        pipeline.optimize_for_gpu()
    
    print("\nGPU optimization test completed!")
    print("\nUsage example:")
    print("pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')")
    print("if torch.cuda.is_available():")
    print("    pipeline.to(torch.device('cuda')).optimize_for_gpu()")
    print("diarization = pipeline('audio.wav')")

if __name__ == "__main__":
    test_gpu_optimization()