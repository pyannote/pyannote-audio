"""VRAM-budget-aware embedding batch selection for speaker diarization.

Replaces the naive "max(64, min(256, free_mb * 0.4 / 60))" auto-scaler with
an empirically-derived ladder. Phase A (2026-04-20) measured diarization
throughput against embedding batch size on RTX A6000 and found that wall
time saturates at batch size 16 (2.2h audio: 103 s @ bs=16, 100 s @ bs=128).
Above bs=16 the pipeline spends an extra 3-7 GB of VRAM for a ~3 % speed
gain. Below bs=4 wall time degrades sharply.

The ladder is therefore capped at 16 as the throughput-optimal point and
steps down to 4 as VRAM tightens. Speaker-count accuracy (measured DER
against pyannote.metrics) is zero across every batch size tried at fp32,
so no accuracy trade-off is involved in picking a smaller batch.

See docs/diarization-vram-profile/README.md in the OpenTranscribe repo for
the raw data.

This module is intentionally pure-Python + stdlib-only so it can be
unit-tested without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Measured 2026-04-20 on RTX A6000 with fork davidamacey/pyannote-audio
# @gpu-optimizations, torch 2.8.0+cu128.
#
# Process footprint (MB) during the diarization "embeddings" stage at a given
# batch size, excluding CUDA context and idle-GPU baseline.
#
# These numbers are the ceiling observed across the small-batch sweep; using
# them for budget decisions is pessimistic on purpose so a budget of exactly
# _DIARIZATION_FOOTPRINT_MB[k] will actually complete.
_DIARIZATION_FOOTPRINT_MB: dict[int, int] = {
    4: 640,
    8: 640,  # bs=8 reuses the torch allocator pool from bs=4
    16: 954,
    32: 1946,  # included for diagnostics; policy never selects bs=32
}

# Absolute throughput ceiling. Phase A found bs=16 saturates wall time.
# Do not raise without re-running the --small-batch-sweep on the target
# hardware; larger batches measured no speed gain and wasted 3-7 GB.
_BATCH_CEILING = 16

# Safety margin subtracted from the caller-provided free_mb to guard against
# allocator fragmentation and small per-call overhead not captured in the
# _DIARIZATION_FOOTPRINT_MB table.
_SAFETY_MARGIN_MB = 200


@dataclass(frozen=True)
class BudgetRecommendation:
    """Output of recommend_embedding_batch.

    Attributes
    ----------
    batch_size : int
        The embedding batch size to assign to
        ``SpeakerDiarization.embedding_batch_size``.
    status : str
        One of ``"optimal"`` (bs=16 fits), ``"tight"`` (bs=4-8 fits), or
        ``"insufficient"`` (even bs=4 does not fit -- caller should fall
        back to CPU embedding or refuse).
    expected_peak_mb : int
        Predicted VRAM footprint at this batch size. Useful for
        reporting and for tests.
    """

    batch_size: int
    status: str
    expected_peak_mb: int


def recommend_embedding_batch(
    free_mb: int,
    *,
    device: str = "cuda",
    ceiling: int = _BATCH_CEILING,
    safety_margin_mb: int = _SAFETY_MARGIN_MB,
) -> BudgetRecommendation:
    """Pick embedding batch size for the given free VRAM budget.

    Parameters
    ----------
    free_mb : int
        Free VRAM available to the diarization process, in megabytes. The
        caller should already have subtracted the CUDA context cost and
        any coexisting model reserves.
    device : str, default ``"cuda"``
        Device type. ``"cuda"``, ``"mps"``, and ``"cpu"`` are accepted.
        On ``"cpu"`` the choice degenerates to ``bs=1`` since the
        constraint is RAM rather than VRAM.
    ceiling : int, default ``_BATCH_CEILING``
        Throughput-optimal ceiling. Do not raise without re-measuring.
    safety_margin_mb : int, default ``_SAFETY_MARGIN_MB``
        Budget withheld for allocator fragmentation.

    Returns
    -------
    BudgetRecommendation

    Notes
    -----
    Called at the start of the embedding stage inside
    :meth:`SpeakerDiarization.apply`. See module docstring for why
    ceiling=16 and why fp32 is the sane default.
    """
    if device == "cpu":
        # CPU path has no VRAM budget; use bs=1 for minimum RAM pressure.
        return BudgetRecommendation(batch_size=1, status="cpu", expected_peak_mb=0)

    headroom = max(0, free_mb - safety_margin_mb)

    # Walk the ladder from the ceiling down; pick the largest entry that fits.
    # "Optimal" means we hit bs=16 (throughput saturation). "Tight" means we
    # had to step down.
    for bs in sorted((k for k in _DIARIZATION_FOOTPRINT_MB if k <= ceiling), reverse=True):
        cost = _DIARIZATION_FOOTPRINT_MB[bs]
        if headroom >= cost:
            status = "optimal" if bs == ceiling else "tight"
            return BudgetRecommendation(
                batch_size=bs, status=status, expected_peak_mb=cost
            )

    # Even bs=4 does not fit.
    return BudgetRecommendation(
        batch_size=4,
        status="insufficient",
        expected_peak_mb=_DIARIZATION_FOOTPRINT_MB[4],
    )


def next_power_of_two_floor(value: int) -> int:
    """Utility: round ``value`` down to the nearest power of two, min 1."""
    if value < 1:
        return 1
    return 2 ** int(math.log2(value))
