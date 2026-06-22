"""ONNX Runtime integration for GPU-optimized inference paths.

Gated by the ``PYANNOTE_USE_ONNX`` environment variable. Default off; when
enabled the segmentation model and WeSpeaker ResNet backbone are served via
``onnxruntime`` instead of eager PyTorch.

Design:

- Segmentation: full waveform → logits path exported as one ONNX graph. Input
  ``(batch, 1, num_samples)`` / output ``(batch, num_frames, num_classes)``.
- Embedding: only the ResNet *backbone* is exported; fbank extraction stays in
  Python and is rewritten to avoid ``torch.vmap`` (which neither TorchScript
  nor ``torch.export`` can trace). Input ``(batch, num_frames, 80)`` fbank
  plus optional ``(batch, num_frames)`` weights → output ``(batch, 256)``.

Provider selection matches the PyTorch device:

- CUDA → ``CUDAExecutionProvider`` (plus optional ``TensorrtExecutionProvider``
  in Phase 6.3).
- MPS → ``CoreMLExecutionProvider`` on Apple Silicon.
- CPU → ``CPUExecutionProvider``.

See ``docs/upstream-patches/phase-6-2-onnx-export-feasibility.md`` in the
OpenTranscribe repository for the full rollout plan and test matrix.
"""

from .runtime import (
    ONNXEmbeddingRuntime,
    ONNXSegmentationRuntime,
    compute_fbank_batched,
    onnx_enabled,
)

__all__ = [
    "ONNXEmbeddingRuntime",
    "ONNXSegmentationRuntime",
    "compute_fbank_batched",
    "onnx_enabled",
]
