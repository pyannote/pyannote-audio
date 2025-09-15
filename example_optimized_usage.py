#!/usr/bin/env python3
"""
Example usage showing the improved pyannote-audio pipeline with better GPU utilization.

This example demonstrates how to use the optimized speaker diarization pipeline
with the fixes for low GPU utilization issues.
"""

import torch
from datetime import datetime, timezone
from pyannote.audio import Pipeline

class OptimizedSpeakerDiarizer:
    """
    Speaker diarization class with optimized GPU utilization.
    
    This class addresses the low GPU utilization issues mentioned in:
    https://github.com/pyannote/pyannote-audio/issues/1403
    """
    
    def __init__(self, pretrained_id="pyannote/speaker-diarization-3.1", device="cuda", 
                 annotator_name="optimized_diarizer", chunk_size_minutes=10, 
                 batch_size=None, use_auth_token=None):
        """
        Initialize the optimized speaker diarization pipeline.
        
        Parameters
        ----------
        pretrained_id : str
            Pretrained model identifier from HuggingFace Hub
        device : str
            Device to use ('cuda' or 'cpu')
        annotator_name : str
            Name for the annotator
        chunk_size_minutes : int
            Chunk size in minutes for processing
        batch_size : int, optional
            Batch size for GPU processing. If None, auto-detects based on GPU memory.
        use_auth_token : str, optional
            HuggingFace authentication token
        """
        
        # Load the pipeline with improved defaults
        self.diarization_pipeline = Pipeline.from_pretrained(
            pretrained_id, use_auth_token=use_auth_token
        )
        
        # Configure for optimal GPU usage
        if torch.cuda.is_available() and device != "cpu":
            self.device = torch.device(device)
            self.diarization_pipeline.to(self.device)
            
            # Apply GPU optimizations
            self.diarization_pipeline.optimize_for_gpu(batch_size=batch_size)
            
            print(f"Pipeline configured for GPU with:")
            print(f"  - Segmentation batch size: {self.diarization_pipeline.segmentation_batch_size}")
            print(f"  - Embedding batch size: {self.diarization_pipeline.embedding_batch_size}")
            print(f"  - Device: {self.device}")
            
        else:
            self.device = torch.device("cpu")
            print("Using CPU for processing")
        
        self.annotator_name = annotator_name
        self.chunk_size_minutes = chunk_size_minutes

    def annotate_whole_audio(self, audio_filepath, turn_length_sec=0.8):
        """
        Perform speaker diarization on the entire audio file.
        
        Parameters
        ----------
        audio_filepath : str
            Path to the audio file
        turn_length_sec : float
            Minimum turn length in seconds
            
        Returns
        -------
        diarization : Annotation
            Speaker diarization results
        """
        creation_time = datetime.now(timezone.utc).isoformat()
        
        print(f"Processing {audio_filepath}...")
        print(f"Started at: {creation_time}")
        
        # The optimized pipeline should now utilize GPU much more efficiently
        diarization = self.diarization_pipeline(audio_filepath)
        
        print(f"Diarization completed. Found {len(diarization.labels())} speakers.")
        
        return diarization

    def get_optimization_info(self):
        """Get information about current optimization settings."""
        info = {
            "device": str(self.device),
            "segmentation_batch_size": getattr(self.diarization_pipeline, 'segmentation_batch_size', 'N/A'),
            "embedding_batch_size": getattr(self.diarization_pipeline, 'embedding_batch_size', 'N/A'),
        }
        
        if torch.cuda.is_available():
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_name"] = torch.cuda.get_device_name(0)
        
        return info

def main():
    """Example usage of the optimized speaker diarization."""
    
    # Example usage showing the improvements
    print("Initializing optimized speaker diarization pipeline...")
    
    # You'll need to set your HuggingFace token or accept conditions at:
    # https://hf.co/pyannote/speaker-diarization-3.1
    diarizer = OptimizedSpeakerDiarizer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=None,  # Auto-detect optimal batch size
        use_auth_token=None  # Replace with your token if needed
    )
    
    # Print optimization info
    print("\nOptimization settings:")
    for key, value in diarizer.get_optimization_info().items():
        print(f"  {key}: {value}")
    
    # Example processing (uncomment and provide a real audio file path)
    # diarization = diarizer.annotate_whole_audio("path/to/your/audio.wav")
    
    print("\nKey improvements implemented:")
    print("1. Default batch sizes increased from 1 to 32 for both segmentation and embedding")
    print("2. Audio file caching to reduce I/O bottlenecks")
    print("3. Auto-detection of optimal batch sizes based on GPU memory")
    print("4. Easy-to-use optimize_for_gpu() method")
    print("5. Better default batch size for overlapped speech detection")

if __name__ == "__main__":
    main()