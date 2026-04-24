"""Phase 5.2 — GPU scatter-reduce port of Inference.aggregate + reconstruct.

Both hot loops in the diarization pipeline (``Inference.aggregate`` in
``core/inference.py`` and ``SpeakerDiarization.reconstruct`` in
``pipelines/speaker_diarization.py``) are scatter-reduction patterns:

- aggregate: per-chunk accumulation of weighted scores into a global frame
  buffer (``index_add_``) plus a running max of presence masks
  (``scatter_reduce_(reduce='amax')``).
- reconstruct: per-(chunk, frame, speaker) cluster-max into a
  ``(num_chunks, num_frames, num_clusters)`` buffer.

CPU vectorization via ``np.add.at`` / ``np.maximum.at`` regressed 35-57% on
the 4.7h benchmark because those ufunc.at primitives disable numpy's
contiguous-stride SIMD. On GPU the story inverts: ``torch.scatter_reduce_``
is a native parallel op with efficient atomic coalescing.

Gating:

- ``PYANNOTE_GPU_AGGREGATE=1`` enables the aggregate GPU path.
- ``PYANNOTE_GPU_RECONSTRUCT=1`` enables the reconstruct GPU path.
- Both default **off** pending DER validation.

Fallback: if the env var is unset OR CUDA is not available OR the allocated
tensors would exceed ``PYANNOTE_GPU_OP_VRAM_BUDGET_MB`` (default 200 MB per
call), control returns to the CPU numpy implementation.

See ``docs/upstream-patches/phase-5-2-gpu-aggregate-and-reconstruct.md`` in
the OpenTranscribe repository for the feasibility memo and expected impact.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def gpu_aggregate_enabled() -> bool:
    return _env_true("PYANNOTE_GPU_AGGREGATE") and torch.cuda.is_available()


def gpu_reconstruct_enabled() -> bool:
    return _env_true("PYANNOTE_GPU_RECONSTRUCT") and torch.cuda.is_available()


def _vram_budget_bytes() -> int:
    mb = int(os.environ.get("PYANNOTE_GPU_OP_VRAM_BUDGET_MB", "200"))
    return mb * 1024 * 1024


def _gpu_device() -> torch.device:
    idx = int(os.environ.get("PYANNOTE_GPU_OP_DEVICE_INDEX", "0"))
    return torch.device(f"cuda:{idx}")


# ---------------------------------------------------------------------------
# aggregate GPU port
# ---------------------------------------------------------------------------


def aggregate_gpu(
    scores_data: np.ndarray,
    start_frames: np.ndarray,
    num_frames: int,
    hamming_window: np.ndarray,
    warm_up_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU scatter-add implementation of the per-chunk aggregate loop.

    Parameters
    ----------
    scores_data
        ``(num_chunks, num_frames_per_chunk, num_classes)`` fp32 (may contain NaN).
    start_frames
        ``(num_chunks,)`` int — per-chunk insertion index into the global buffer.
    num_frames
        Total frames in the aggregated output.
    hamming_window, warm_up_window
        ``(num_frames_per_chunk, 1)`` fp32 weights applied per chunk.

    Returns
    -------
    aggregated_output
        ``(num_frames, num_classes)`` fp32 — weighted-score sum per frame.
    overlapping_chunk_count
        ``(num_frames, num_classes)`` fp32 — per-frame weighted mask sum (denom).
    aggregated_mask
        ``(num_frames, num_classes)`` fp32 {0, 1} — at least one non-NaN frame.
    """
    C, F, K = scores_data.shape
    device = _gpu_device()

    # Estimated peak VRAM: 3 × (C, F, K) fp32 tensors + indices.
    bytes_needed = 3 * C * F * K * 4 + 2 * num_frames * K * 4 + C * F * 8
    if bytes_needed > _vram_budget_bytes():
        raise _VRAMExceeded(bytes_needed)

    scores_t = torch.from_numpy(
        np.ascontiguousarray(scores_data, dtype=np.float32)
    ).to(device, non_blocking=True)
    hamming_t = torch.from_numpy(
        np.ascontiguousarray(hamming_window, dtype=np.float32)
    ).to(device, non_blocking=True)  # (F, 1)
    warm_up_t = torch.from_numpy(
        np.ascontiguousarray(warm_up_window, dtype=np.float32)
    ).to(device, non_blocking=True)  # (F, 1)
    start_t = torch.from_numpy(
        np.ascontiguousarray(start_frames, dtype=np.int64)
    ).to(device, non_blocking=True)  # (C,)

    mask_t = (~torch.isnan(scores_t)).float()  # (C, F, K)
    score_clean = torch.where(torch.isnan(scores_t), torch.zeros_like(scores_t), scores_t)

    weight_2d = (hamming_t * warm_up_t)  # (F, 1)  broadcasts to (C, F, K)
    weighted_score = score_clean * mask_t * weight_2d
    weighted_mask = mask_t * weight_2d
    mask_presence = mask_t

    # Flatten to (C*F, K) source + (C*F,) destination indices
    offsets = torch.arange(F, device=device, dtype=torch.int64)
    global_idx = (start_t[:, None] + offsets[None, :]).reshape(-1)  # (C*F,)
    assert global_idx.max().item() < num_frames, (
        f"aggregate_gpu: index out of bounds, max={global_idx.max().item()} "
        f"num_frames={num_frames}"
    )

    agg_out = torch.zeros((num_frames, K), device=device, dtype=torch.float32)
    agg_out.index_add_(0, global_idx, weighted_score.reshape(-1, K))

    overlap = torch.zeros((num_frames, K), device=device, dtype=torch.float32)
    overlap.index_add_(0, global_idx, weighted_mask.reshape(-1, K))

    # scatter_reduce_ with amax needs (flat_len, K) source and broadcast index
    agg_mask = torch.zeros((num_frames, K), device=device, dtype=torch.float32)
    agg_mask.scatter_reduce_(
        0,
        global_idx.unsqueeze(1).expand(-1, K),
        mask_presence.reshape(-1, K),
        reduce="amax",
        include_self=True,
    )

    return (
        agg_out.cpu().numpy(),
        overlap.cpu().numpy(),
        agg_mask.cpu().numpy(),
    )


# ---------------------------------------------------------------------------
# reconstruct GPU port
# ---------------------------------------------------------------------------


def reconstruct_gpu(
    seg_data: np.ndarray,
    hard_clusters: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """GPU scatter-max implementation of the per-(chunk, cluster) reconstruction.

    Parameters
    ----------
    seg_data
        ``(num_chunks, num_frames, local_num_speakers)`` fp32 segmentation.
    hard_clusters
        ``(num_chunks, local_num_speakers)`` int cluster assignment.
        Values in ``[0, num_clusters)`` or ``-2`` for inactive speakers.
    num_clusters
        Total cluster count (``max(hard_clusters) + 1``).

    Returns
    -------
    clustered_segmentations
        ``(num_chunks, num_frames, num_clusters)`` fp32 — max over local
        speakers grouped by cluster; NaN for clusters not present in a chunk.
    """
    C, F, S = seg_data.shape
    device = _gpu_device()

    # Estimated peak VRAM
    bytes_needed = (
        C * F * S * 4              # seg_t
        + C * F * num_clusters * 4 # out_flat float
        + C * F * S * 8            # flat_idx int64
        + C * F * S * 4            # values (float)
    )
    if bytes_needed > _vram_budget_bytes():
        raise _VRAMExceeded(bytes_needed)

    seg_t = torch.from_numpy(
        np.ascontiguousarray(seg_data, dtype=np.float32)
    ).to(device, non_blocking=True)
    hc_t = torch.from_numpy(
        np.ascontiguousarray(hard_clusters, dtype=np.int64)
    ).to(device, non_blocking=True)

    valid = hc_t >= 0            # (C, S)
    clamped = hc_t.clamp_min(0)  # (C, S) — safe for index, masked via valid

    # Build index tensor: for each (c, f, s) the flat position in
    # (C, F, num_clusters).stride = c * F * NC + f * NC + clamped[c, s]
    c_idx = torch.arange(C, device=device, dtype=torch.int64)[:, None, None]
    f_idx = torch.arange(F, device=device, dtype=torch.int64)[None, :, None]
    # broadcast to (C, F, S)
    k_idx = clamped[:, None, :].expand(C, F, S)
    valid_b = valid[:, None, :].expand(C, F, S)

    flat_idx = (c_idx * (F * num_clusters) + f_idx * num_clusters + k_idx).reshape(-1)
    neg_inf = torch.finfo(torch.float32).min
    values = torch.where(
        valid_b, seg_t, torch.full_like(seg_t, neg_inf)
    ).reshape(-1)

    out_flat = torch.full(
        (C * F * num_clusters,), neg_inf, device=device, dtype=torch.float32,
    )
    out_flat.scatter_reduce_(
        0, flat_idx, values, reduce="amax", include_self=True,
    )

    out = out_flat.view(C, F, num_clusters)
    # Fill slots that never received a valid value with NaN (original behavior)
    result = torch.where(
        out <= neg_inf / 2,
        torch.full_like(out, float("nan")),
        out,
    )
    return result.cpu().numpy()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _VRAMExceeded(RuntimeError):
    """Raised internally to signal GPU-path budget exhaustion; caught by caller."""

    def __init__(self, bytes_needed: int):
        super().__init__(f"GPU op would need {bytes_needed / 1024 / 1024:.0f} MB")
        self.bytes_needed = bytes_needed


def try_aggregate_gpu(
    scores_data: np.ndarray,
    start_frames: np.ndarray,
    num_frames: int,
    hamming_window: np.ndarray,
    warm_up_window: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Best-effort GPU aggregate; ``None`` when disabled or budget-blocked."""
    if not gpu_aggregate_enabled():
        return None
    try:
        return aggregate_gpu(
            scores_data, start_frames, num_frames, hamming_window, warm_up_window
        )
    except (_VRAMExceeded, RuntimeError):
        return None


def try_reconstruct_gpu(
    seg_data: np.ndarray,
    hard_clusters: np.ndarray,
    num_clusters: int,
) -> Optional[np.ndarray]:
    """Best-effort GPU reconstruct; ``None`` when disabled or budget-blocked."""
    if not gpu_reconstruct_enabled():
        return None
    try:
        return reconstruct_gpu(seg_data, hard_clusters, num_clusters)
    except (_VRAMExceeded, RuntimeError):
        return None
