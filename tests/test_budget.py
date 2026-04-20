"""Unit tests for pyannote.audio.pipelines._budget.

Phase D.1 of the VRAM-budget plan. Pure-Python, no GPU. Asserts the
measured Phase A ladder translates correctly into BudgetRecommendation
instances.
"""

from __future__ import annotations

import pytest

from pyannote.audio.pipelines._budget import (
    _BATCH_CEILING,
    _DIARIZATION_FOOTPRINT_MB,
    _DIARIZATION_FOOTPRINT_MB_BY_DEVICE,
    _SAFETY_MARGIN_MB,
    BudgetRecommendation,
    _footprint_table,
    next_power_of_two_floor,
    recommend_embedding_batch,
)


class TestRecommendEmbeddingBatch:
    """Budget ladder is capped at bs=16 and steps down through {8, 4}."""

    def test_abundant_vram_returns_optimal_ceiling(self) -> None:
        # 48 GB GPU -- far more than enough for every batch in the ladder.
        rec = recommend_embedding_batch(free_mb=48_000)
        assert rec.batch_size == _BATCH_CEILING
        assert rec.status == "optimal"
        assert rec.expected_peak_mb == _DIARIZATION_FOOTPRINT_MB[16]

    def test_exact_threshold_for_bs16_still_fits(self) -> None:
        # footprint(16) + safety_margin. One MB below this would drop to bs=8.
        need = _DIARIZATION_FOOTPRINT_MB[16] + _SAFETY_MARGIN_MB
        rec = recommend_embedding_batch(free_mb=need)
        assert rec.batch_size == 16
        assert rec.status == "optimal"

    def test_just_below_bs16_threshold_picks_bs8(self) -> None:
        need = _DIARIZATION_FOOTPRINT_MB[16] + _SAFETY_MARGIN_MB - 1
        rec = recommend_embedding_batch(free_mb=need)
        assert rec.batch_size == 8
        assert rec.status == "tight"

    def test_4gb_laptop_recipe_fits_bs8(self) -> None:
        # 4 GB card with 500 MB CUDA context + 500 MB other tasks =
        # ~3 GB available for diarization. More than enough for bs=8.
        rec = recommend_embedding_batch(free_mb=3_000)
        # Measured Phase A finding: at this budget we still get the ceiling.
        assert rec.batch_size == 16
        assert rec.status == "optimal"

    def test_tight_4gb_shared_with_whisper_picks_bs8(self) -> None:
        # 4 GB minus baseline minus CUDA context minus Whisper small (750)
        # leaves ~800 MB -- below bs=16 threshold (954 + 200) but above bs=8.
        rec = recommend_embedding_batch(free_mb=850)
        assert rec.batch_size == 8
        assert rec.status == "tight"

    def test_below_bs8_threshold_returns_insufficient(self) -> None:
        # Phase A measurement: bs=4 and bs=8 share the 640 MB torch allocator
        # pool, so there is no intermediate "bs=4 tight" regime. Below
        # 640 + safety_margin the policy drops to "insufficient" and the
        # caller must decide (CPU fallback or refuse).
        rec = recommend_embedding_batch(free_mb=830)
        assert rec.batch_size == 4
        assert rec.status == "insufficient"

    def test_insufficient_budget_returns_bs4_with_status(self) -> None:
        # Less than even bs=4 needs: policy still returns bs=4 (caller
        # decides whether to fall back to CPU) and signals insufficient.
        rec = recommend_embedding_batch(free_mb=100)
        assert rec.batch_size == 4
        assert rec.status == "insufficient"

    def test_zero_budget_returns_insufficient(self) -> None:
        rec = recommend_embedding_batch(free_mb=0)
        assert rec.status == "insufficient"

    def test_negative_budget_treated_as_zero(self) -> None:
        # Caller may pass a negative after subtracting reserves; we clamp.
        rec = recommend_embedding_batch(free_mb=-500)
        assert rec.status == "insufficient"

    def test_cpu_device_returns_batch_1(self) -> None:
        rec = recommend_embedding_batch(free_mb=999_999, device="cpu")
        assert rec.batch_size == 1
        assert rec.status == "cpu"

    def test_mps_device_uses_vram_ladder(self) -> None:
        # MPS budget comes from the unified memory pool but the ladder
        # shape is the same (capped at bs=16). Footprint values differ
        # per device because allocator behaviour differs.
        rec = recommend_embedding_batch(free_mb=5_000, device="mps")
        assert rec.batch_size == 16
        assert rec.status == "optimal"
        # MPS pre-reserves ~1.9 GB at bs <= 16 (measured Mac Studio M2 Max)
        assert rec.expected_peak_mb == 1900

    def test_mps_footprint_is_higher_than_cuda(self) -> None:
        # 1500 MB free: on CUDA this fits bs=16 (~950 MB cost + 200 margin);
        # on MPS it does not (~1900 MB pre-reservation + 200 margin = 2100).
        cuda_rec = recommend_embedding_batch(free_mb=1500, device="cuda")
        mps_rec = recommend_embedding_batch(free_mb=1500, device="mps")
        assert cuda_rec.batch_size == 16
        assert mps_rec.batch_size == 4  # must step down on MPS
        assert mps_rec.status == "insufficient"

    def test_mps_at_2500mb_fits_bs16(self) -> None:
        # MPS needs 1900 + 200 margin = 2100 MB minimum for bs=16.
        rec = recommend_embedding_batch(free_mb=2_500, device="mps")
        assert rec.batch_size == 16
        assert rec.status == "optimal"

    def test_unknown_device_falls_back_to_cuda_table(self) -> None:
        # Fallback for devices we haven't characterized yet.
        rec = recommend_embedding_batch(free_mb=1500, device="xpu")
        assert rec.batch_size == 16  # uses CUDA footprint table
        assert rec.status == "optimal"

    def test_cuda_indexed_device_strips_index(self) -> None:
        # torch.device spec like "cuda:0" should match the "cuda" table.
        rec = recommend_embedding_batch(free_mb=1500, device="cuda:0")
        assert rec.batch_size == 16

    def test_custom_ceiling_honored(self) -> None:
        # Caller explicitly caps lower than default (e.g. for tests).
        rec = recommend_embedding_batch(free_mb=48_000, ceiling=8)
        assert rec.batch_size == 8
        assert rec.status == "optimal"

    def test_custom_safety_margin_honored(self) -> None:
        need = _DIARIZATION_FOOTPRINT_MB[16]  # no margin
        rec = recommend_embedding_batch(free_mb=need, safety_margin_mb=0)
        assert rec.batch_size == 16
        assert rec.status == "optimal"

    def test_returned_type_is_frozen_dataclass(self) -> None:
        rec = recommend_embedding_batch(free_mb=10_000)
        assert isinstance(rec, BudgetRecommendation)
        with pytest.raises((AttributeError, Exception)):
            rec.batch_size = 999  # type: ignore[misc]


class TestNextPowerOfTwoFloor:
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (7, 4),
        (8, 8),
        (31, 16),
        (32, 32),
        (100, 64),
        (256, 256),
    ])
    def test_exact_cases(self, value: int, expected: int) -> None:
        assert next_power_of_two_floor(value) == expected

    def test_zero_and_negative_clamp_to_one(self) -> None:
        assert next_power_of_two_floor(0) == 1
        assert next_power_of_two_floor(-10) == 1


class TestBudgetLadderShape:
    """Regression: the ladder tables must match measurements."""

    def test_ladder_keys_cover_policy_range(self) -> None:
        # Must have entries for every batch the recommender may return.
        for device in ("cuda", "mps"):
            table = _DIARIZATION_FOOTPRINT_MB_BY_DEVICE[device]
            assert 4 in table, f"missing bs=4 on {device}"
            assert 8 in table, f"missing bs=8 on {device}"
            assert 16 in table, f"missing bs=16 on {device}"

    def test_footprint_monotonic_non_decreasing(self) -> None:
        # Larger batch never costs less VRAM than smaller.
        for device, table in _DIARIZATION_FOOTPRINT_MB_BY_DEVICE.items():
            entries = sorted(table.items())
            for (_, a), (_, b) in zip(entries[:-1], entries[1:]):
                assert b >= a, f"non-monotonic footprint on {device}: {entries}"

    def test_ceiling_is_in_every_device_table(self) -> None:
        for device, table in _DIARIZATION_FOOTPRINT_MB_BY_DEVICE.items():
            assert _BATCH_CEILING in table, f"missing ceiling on {device}"

    def test_footprint_table_helper_resolves_devices(self) -> None:
        assert _footprint_table("cuda") is _DIARIZATION_FOOTPRINT_MB_BY_DEVICE["cuda"]
        assert _footprint_table("mps") is _DIARIZATION_FOOTPRINT_MB_BY_DEVICE["mps"]
        # Case-insensitive + index-stripping:
        assert _footprint_table("CUDA:0") is _DIARIZATION_FOOTPRINT_MB_BY_DEVICE["cuda"]
        # Unknown device returns the cuda fallback:
        assert _footprint_table("xpu") is _DIARIZATION_FOOTPRINT_MB_BY_DEVICE["cuda"]
