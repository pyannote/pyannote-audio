"""Runtime wrappers around ONNX Runtime sessions for the fork's hot models.

These classes are designed to drop into the same call sites as eager PyTorch
inference with minimal branching in the pipeline. Provider selection is
device-aware; all wrappers accept torch tensors in / return torch tensors out
on the original device.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi


def onnx_enabled() -> bool:
    """Global flag: honor ``PYANNOTE_USE_ONNX`` env var, 1/true/yes → on."""
    val = os.environ.get("PYANNOTE_USE_ONNX", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _onnx_models_dir() -> Optional[Path]:
    """Return the directory holding exported .onnx artifacts (or None)."""
    raw = os.environ.get("PYANNOTE_ONNX_MODELS_DIR")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


def _shapes_to_str(shape_map: Optional[dict]) -> Optional[str]:
    """Convert ``{input_name: (d0, d1, ...)}`` → ORT's comma/x-joined string.

    Example: ``{"input": (1, 1, 80000)}`` → ``"input:1x1x80000"``. Multiple
    inputs become comma-separated: ``"a:2x3,b:4x5"``.
    """
    if not shape_map:
        return None
    return ",".join(
        f"{name}:{'x'.join(str(d) for d in shape)}"
        for name, shape in shape_map.items()
    )


def _select_providers(
    device: torch.device,
    shape_profile: Optional[dict] = None,
) -> list:
    """Map a torch device to an ORT provider list with safe fallbacks.

    Parameters
    ----------
    device
        Target device; drives provider selection.
    shape_profile
        Optional ``{"min": {name: shape}, "opt": {name: shape}, "max": {...}}``
        for TRT EP's engine-plan builder. Without a shape profile, TRT EP
        rebuilds the plan on every distinct input shape it sees — catastrophic
        for pyannote which emits variable batch sizes.
    """
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        providers: list = []
        if os.environ.get("ENABLE_TENSORRT", "").strip().lower() in {"1", "true", "yes", "on"}:
            trt_cache = os.environ.get(
                "TENSORRT_CACHE_DIR",
                str(Path.home() / ".cache" / "tensorrt" / f"sm_{torch.cuda.get_device_capability(idx)[0]}{torch.cuda.get_device_capability(idx)[1]}"),
            )
            Path(trt_cache).mkdir(parents=True, exist_ok=True)
            trt_opts: dict = {
                "device_id": idx,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
                "trt_fp16_enable": False,  # fp32-only per DER invariant
                "trt_max_workspace_size": 2 << 30,
            }
            if shape_profile:
                min_s = _shapes_to_str(shape_profile.get("min"))
                opt_s = _shapes_to_str(shape_profile.get("opt"))
                max_s = _shapes_to_str(shape_profile.get("max"))
                if min_s and opt_s and max_s:
                    trt_opts["trt_profile_min_shapes"] = min_s
                    trt_opts["trt_profile_opt_shapes"] = opt_s
                    trt_opts["trt_profile_max_shapes"] = max_s
            providers.append(("TensorrtExecutionProvider", trt_opts))
        providers.append(("CUDAExecutionProvider", {"device_id": idx}))
        providers.append("CPUExecutionProvider")
        return providers
    if device.type == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# Shape profiles for the two ONNX graphs the pipeline exports. Values chosen
# to bracket the shapes the real pipeline emits on a 10s chunk @ 16 kHz with
# embedding_batch_size=16 (Phase A pin) and segmentation_batch_size=32.

_SEGMENTATION_SHAPE_PROFILE = {
    "min": {"input_values": (1, 1, 80000)},
    "opt": {"input_values": (32, 1, 80000)},
    "max": {"input_values": (32, 1, 160000)},
}

_EMBEDDING_SHAPE_PROFILE = {
    "min": {"fbank_features": (1, 50, 80), "weights": (1, 50)},
    "opt": {"fbank_features": (16, 200, 80), "weights": (16, 200)},
    "max": {"fbank_features": (32, 500, 80), "weights": (32, 500)},
}


# ---------------------------------------------------------------------------
# Batched fbank: replaces torch.vmap(self._fbank) from WeSpeakerResNet34
# ---------------------------------------------------------------------------


def compute_fbank_batched(
    waveforms: torch.Tensor,
    fbank_fn,
    fbank_centering_span: Optional[float] = None,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> torch.Tensor:
    """Batched fbank extraction without ``torch.vmap``.

    Parameters
    ----------
    waveforms
        ``(batch, 1, num_samples)`` mono audio at ``sample_rate``.
    fbank_fn
        ``functools.partial`` wrapping ``torchaudio.compliance.kaldi.fbank``
        with its static keyword arguments bound.
    fbank_centering_span
        If set, use running-average centering over this many seconds instead
        of per-utterance global average.
    sample_rate, frame_length_ms, frame_shift_ms
        Must match the values baked into ``fbank_fn`` — reused here only to
        recompute the running-average kernel size.

    Returns
    -------
    torch.Tensor
        ``(batch, num_frames, num_mel_bins)`` — identical output to the
        vmap-based implementation in ``WeSpeakerResNet34.compute_fbank``.

    Notes
    -----
    The vmap-based original does an implicit batched FFT. This batched loop
    calls ``fbank_fn`` once per sample. For the typical embedding_batch_size=16
    configuration the cost is negligible (~200 µs/sample on CPU, <50 µs on
    GPU) and dominated by the subsequent ResNet forward. The explicit loop is
    ONNX-traceable where vmap is not.
    """
    waveforms = waveforms * (1 << 15)
    device = waveforms.device
    # Kaldi fbank runs on host; MPS fbank computed on CPU then pushed back.
    # CUDA path keeps tensors on device — torchaudio.compliance.kaldi.fbank
    # honours input device for fp32 inputs.
    if device.type == "mps":
        src = waveforms.cpu()
    else:
        src = waveforms

    per_sample = [fbank_fn(src[b]) for b in range(src.shape[0])]
    features = torch.stack(per_sample, dim=0)
    if device.type == "mps":
        features = features.to(device)

    if fbank_centering_span is None:
        return features - torch.mean(features, dim=1, keepdim=True)

    # Running-average centering — same math as the upstream implementation.
    window_size = int(sample_rate * frame_length_ms * 0.001)
    step_size = int(sample_rate * frame_shift_ms * 0.001)
    span_samples = int(fbank_centering_span * sample_rate)
    # conv1d_num_frames would be ideal but we keep the helper self-contained.
    kernel_frames = max(
        1, (span_samples - window_size) // step_size + 1
    )
    return features - F.avg_pool1d(
        features.transpose(1, 2),
        kernel_size=2 * (kernel_frames // 2) + 1,
        stride=1,
        padding=kernel_frames // 2,
        count_include_pad=False,
    ).transpose(1, 2)


# ---------------------------------------------------------------------------
# Segmentation ONNX runtime
# ---------------------------------------------------------------------------


class ONNXSegmentationRuntime:
    """ORT session for pyannote segmentation-3.0.

    Input:  waveform  ``(batch, 1, num_samples)``  @ 16 kHz, fp32.
    Output: logits    ``(batch, num_frames, num_classes)``  fp32 (pre-softmax).
    """

    def __init__(self, onnx_path: Path, device: torch.device):
        import onnxruntime as ort
        self._path = Path(onnx_path)
        self._device = device
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # warnings+
        self.session = ort.InferenceSession(
            str(self._path), sess_options=sess_options,
            providers=_select_providers(device, shape_profile=_SEGMENTATION_SHAPE_PROFILE),
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        arr = waveforms.detach().contiguous().float()
        if arr.device.type != "cpu":
            arr = arr.cpu()
        out_np = self.session.run(
            [self._output_name], {self._input_name: arr.numpy()}
        )[0]
        return torch.from_numpy(out_np).to(waveforms.device)

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())


# ---------------------------------------------------------------------------
# Embedding ONNX runtime (ResNet backbone only — no fbank inside graph)
# ---------------------------------------------------------------------------


class ONNXEmbeddingRuntime:
    """ORT session for the WeSpeaker ResNet backbone.

    Input:   fbank    ``(batch, num_frames, 80)``
             weights  ``(batch, num_frames)`` — optional, passes-through zeros
                       when omitted.
    Output:  embedding ``(batch, 256)`` fp32.
    """

    def __init__(self, onnx_path: Path, device: torch.device):
        import onnxruntime as ort
        self._path = Path(onnx_path)
        self._device = device
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self._path), sess_options=sess_options,
            providers=_select_providers(device, shape_profile=_EMBEDDING_SHAPE_PROFILE),
        )
        self._inputs = [i.name for i in self.session.get_inputs()]
        self._output_name = self.session.get_outputs()[0].name

    def __call__(
        self,
        fbank: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        arr_fb = fbank.detach().contiguous().float()
        if arr_fb.device.type != "cpu":
            arr_fb = arr_fb.cpu()
        feed = {self._inputs[0]: arr_fb.numpy()}
        if len(self._inputs) > 1:
            if weights is None:
                weights = torch.ones(
                    fbank.shape[0], fbank.shape[1],
                    dtype=fbank.dtype, device=fbank.device,
                )
            arr_w = weights.detach().contiguous().float()
            if arr_w.device.type != "cpu":
                arr_w = arr_w.cpu()
            feed[self._inputs[1]] = arr_w.numpy()
        out_np = self.session.run([self._output_name], feed)[0]
        return torch.from_numpy(out_np).to(fbank.device)

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())


# ---------------------------------------------------------------------------
# Convenience loader — resolves artifacts via PYANNOTE_ONNX_MODELS_DIR
# ---------------------------------------------------------------------------


def load_default_runtimes(
    device: torch.device,
) -> tuple[Optional[ONNXSegmentationRuntime], Optional[ONNXEmbeddingRuntime]]:
    """Best-effort load of both runtimes from the configured artifacts dir.

    Returns a tuple ``(seg, emb)`` where each may be ``None`` if the artifact
    is missing or ORT session construction failed. Callers must handle the
    ``None`` case by falling back to eager PyTorch.
    """
    models_dir = _onnx_models_dir()
    if models_dir is None:
        return None, None

    seg_path = models_dir / "segmentation.onnx"
    emb_path = models_dir / "embedding.onnx"

    seg = None
    emb = None
    if seg_path.exists():
        try:
            seg = ONNXSegmentationRuntime(seg_path, device)
        except Exception:
            seg = None
    if emb_path.exists():
        try:
            emb = ONNXEmbeddingRuntime(emb_path, device)
        except Exception:
            emb = None
    return seg, emb
