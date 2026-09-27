"""Export the fork's hot models to ONNX.

Two graphs are produced:

- ``segmentation.onnx`` — pyannote/segmentation-3.0 end-to-end (waveform → logits)
- ``embedding.onnx``   — WeSpeaker ResNet backbone (fbank → embedding)

Invoke from the command line:

    python -m pyannote.audio.onnx.export --out-dir /tmp/onnx --token $HF_TOKEN

or from Python:

    from pyannote.audio.onnx.export import export_segmentation, export_embedding

The embedding export follows pyannote-audio discussion #1929: the outer
``compute_fbank`` path uses ``torch.vmap`` which is not traceable, so only the
ResNet backbone is exported. Fbank extraction is replaced at runtime by
``onnx.runtime.compute_fbank_batched``.

Every produced ``.onnx`` has a sibling ``<name>.metadata.json`` recording the
fork SHA, torch / onnx / onnxruntime versions, HF model revision, and SHA-256
of the ``.onnx`` bytes. This metadata is what the runtime uses to detect cache
invalidation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fork_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_metadata(path: Path, **fields) -> None:
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
    except ImportError:
        ort_version = "unavailable"
    try:
        import onnx
        onnx_version = onnx.__version__
    except ImportError:
        onnx_version = "unavailable"

    meta = {
        "fork_sha": _fork_sha(),
        "torch_version": torch.__version__,
        "onnx_version": onnx_version,
        "onnxruntime_version": ort_version,
        "onnx_bytes_sha256": _sha256(path),
        "onnx_size_bytes": path.stat().st_size,
        **fields,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def _force_eval(module: torch.nn.Module) -> None:
    """Recursively set every submodule to eval mode (even leaf flags)."""
    module.eval()
    for mod in module.modules():
        mod.eval()
        if hasattr(mod, "training"):
            mod.training = False


# ---------------------------------------------------------------------------
# Segmentation export
# ---------------------------------------------------------------------------


def export_segmentation(
    out_path: Path,
    *,
    model_id: str = "pyannote/segmentation-3.0",
    token: Optional[str] = None,
    opset_version: int = 18,
    device: str = "cpu",
) -> Path:
    """Export the end-to-end segmentation model.

    Follows the ``onnx-community/pyannote-segmentation-3.0`` recipe:
    ``do_constant_folding=True``, ``input_values`` / ``logits`` names, dynamic
    axes on batch / channel / sample / frame dimensions. The spike verified
    that post-softmax argmax is frame-class bit-identical vs PyTorch eager
    despite a 5.6e-3 raw-logit drift.
    """
    from pyannote.audio import Model

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = Model.from_pretrained(model_id, token=token).to(device)
    _force_eval(model)

    dummy = torch.zeros(2, 1, 160000, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        opset_version=opset_version,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
            "logits": {0: "batch_size", 1: "num_frames"},
        },
        do_constant_folding=True,
    )
    _write_metadata(
        out_path,
        model="segmentation",
        hf_model_id=model_id,
        input_schema={"input_values": "(B, C, S) float32 @ 16 kHz"},
        output_schema={"logits": "(B, num_frames, 7) float32 pre-softmax"},
    )
    return out_path


# ---------------------------------------------------------------------------
# Embedding export — backbone only (no fbank inside graph)
# ---------------------------------------------------------------------------


class _EmbeddingBackboneWrapper(torch.nn.Module):
    """Exports only the ResNet backbone with both fbank + weights inputs.

    Returns ``embed_b`` (second of two-emb-layer outputs) to match the
    production pipeline's ``resnet(fbank, weights=imasks)[1]`` usage.
    """

    def __init__(self, resnet: torch.nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        out = self.resnet(fbank, weights=weights)
        if isinstance(out, tuple):
            return out[-1]
        return out


def export_embedding(
    out_path: Path,
    *,
    model_id: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
    token: Optional[str] = None,
    opset_version: int = 18,
    device: str = "cpu",
    num_frames: int = 200,
    num_mels: int = 80,
) -> Path:
    """Export the WeSpeaker ResNet backbone as ONNX.

    The backbone accepts pre-computed fbank features + per-frame weights and
    returns ``(batch, 256)`` embeddings. Fbank extraction is NOT in the graph
    — the runtime wrapper computes fbank in Python via
    ``compute_fbank_batched`` (vmap-free).
    """
    from pyannote.audio import Model

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = Model.from_pretrained(model_id, token=token).to(device)
    _force_eval(model)

    backbone = _EmbeddingBackboneWrapper(model.resnet).to(device)
    _force_eval(backbone)

    fbank = torch.randn(2, num_frames, num_mels, device=device)
    weights = torch.ones(2, num_frames, device=device)

    torch.onnx.export(
        backbone,
        (fbank, weights),
        str(out_path),
        opset_version=opset_version,
        input_names=["fbank_features", "weights"],
        output_names=["embeddings"],
        dynamic_axes={
            "fbank_features": {0: "batch_size", 1: "num_frames"},
            "weights": {0: "batch_size", 1: "num_frames"},
            "embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    _write_metadata(
        out_path,
        model="embedding",
        hf_model_id=model_id,
        input_schema={
            "fbank_features": "(B, num_frames, 80) float32",
            "weights": "(B, num_frames) float32",
        },
        output_schema={"embeddings": "(B, 256) float32"},
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="Directory to write segmentation.onnx + embedding.onnx",
    )
    parser.add_argument(
        "--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="HF access token (default: HF_TOKEN / HUGGINGFACE_TOKEN env)",
    )
    parser.add_argument("--only", choices=["seg", "emb", "both"], default="both")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("seg", "both"):
        p = export_segmentation(
            out_dir / "segmentation.onnx",
            token=args.token,
            opset_version=args.opset,
            device=args.device,
        )
        print(f"✓ segmentation → {p} ({p.stat().st_size // 1024} KiB)")
    if args.only in ("emb", "both"):
        p = export_embedding(
            out_dir / "embedding.onnx",
            token=args.token,
            opset_version=args.opset,
            device=args.device,
        )
        print(f"✓ embedding   → {p} ({p.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
