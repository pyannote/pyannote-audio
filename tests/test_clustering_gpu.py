"""Numerical-parity + VRAM-budget tests for the Phase 3 GPU clustering helpers.

These exercise `_gpu_cdist`, `_gpu_pdist_condensed`, and the
`_gpu_clustering_fits` / `_gpu_clustering_budget_bytes` guard against
`scipy.spatial.distance.cdist` / `scipy.spatial.distance.pdist` references on
fixed-seed synthetic inputs. The GPU path is exercised when a CUDA or MPS
device is present; otherwise the tests fall through the scipy path and verify
the fallback still produces scipy-equivalent output.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.spatial.distance import pdist as scipy_pdist

from pyannote.audio.pipelines.clustering import (
    _get_clustering_device,
    _gpu_cdist,
    _gpu_clustering_budget_bytes,
    _gpu_clustering_fits,
    _gpu_pdist_condensed,
)


RTOL = 1e-4  # tolerance — fp32 on GPU vs fp64 scipy
ATOL = 1e-5


def _make_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x


def _has_gpu() -> bool:
    if torch.cuda.is_available():
        return True
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


class TestGPUCdist:
    """Pairwise distance: `_gpu_cdist` vs scipy.cdist."""

    @pytest.mark.parametrize("n_a,n_b,dim", [(50, 50, 256), (500, 20, 256), (100, 100, 512)])
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_parity_small(self, n_a: int, n_b: int, dim: int, metric: str) -> None:
        a = _make_embeddings(n_a, dim, seed=1)
        b = _make_embeddings(n_b, dim, seed=2)
        device = _get_clustering_device()
        gpu_result = _gpu_cdist(a, b, metric, device)
        cpu_result = scipy_cdist(a, b, metric=metric)
        assert gpu_result.shape == cpu_result.shape
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=RTOL, atol=ATOL)

    def test_highly_similar_vectors(self) -> None:
        """Numerical stability on near-identical vectors (dot product ~ 1)."""
        base = _make_embeddings(100, 256, seed=10)
        perturbed = base + 1e-5 * _make_embeddings(100, 256, seed=11)
        device = _get_clustering_device()
        gpu = _gpu_cdist(base, perturbed, "cosine", device)
        cpu = scipy_cdist(base, perturbed, metric="cosine")
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)

    def test_asymmetric_shape(self) -> None:
        a = _make_embeddings(20, 128, seed=3)
        b = _make_embeddings(50, 128, seed=4)
        device = _get_clustering_device()
        result = _gpu_cdist(a, b, "cosine", device)
        assert result.shape == (20, 50)

    def test_unsupported_metric_falls_back(self) -> None:
        """Metrics without a fast GPU path route through scipy."""
        a = _make_embeddings(10, 16, seed=5)
        b = _make_embeddings(10, 16, seed=6)
        device = _get_clustering_device()
        gpu = _gpu_cdist(a, b, "hamming", device)
        cpu = scipy_cdist(a, b, metric="hamming")
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)


class TestGPUPdistCondensed:
    """Condensed pdist: `_gpu_pdist_condensed` vs scipy.pdist."""

    @pytest.mark.parametrize("n,dim", [(50, 256), (500, 256), (100, 512)])
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_parity(self, n: int, dim: int, metric: str) -> None:
        a = _make_embeddings(n, dim, seed=7)
        device = _get_clustering_device()
        gpu = _gpu_pdist_condensed(a, metric, device)
        cpu = scipy_pdist(a, metric=metric)
        assert gpu.shape == cpu.shape == (n * (n - 1) // 2,)
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)

    def test_single_element_returns_empty(self) -> None:
        """pdist on N=1 should produce an empty condensed vector."""
        a = _make_embeddings(1, 16, seed=9)
        device = _get_clustering_device()
        gpu = _gpu_pdist_condensed(a, "cosine", device)
        cpu = scipy_pdist(a, metric="cosine")
        assert gpu.shape == cpu.shape == (0,)


class TestBudgetGuard:
    """Memory-budget guard must fall back to scipy rather than OOM."""

    def test_cpu_always_falls_back(self) -> None:
        cpu = torch.device("cpu")
        assert not _gpu_clustering_fits(100, 256, cpu)
        assert _gpu_clustering_budget_bytes(cpu) == 0

    @pytest.mark.skipif(not _has_gpu(), reason="no GPU available for budget probing")
    def test_small_n_fits_gpu(self) -> None:
        device = _get_clustering_device()
        if device.type == "cpu":
            pytest.skip("auto-detect returned CPU")
        assert _gpu_clustering_fits(500, 256, device)

    def test_tiny_budget_forces_fallback(self, monkeypatch) -> None:
        """Setting the budget to 1 MB makes every non-trivial call fall back."""
        monkeypatch.setenv("PYANNOTE_CLUSTERING_VRAM_BUDGET_MB", "1")
        a = _make_embeddings(500, 256, seed=12)
        b = _make_embeddings(500, 256, seed=13)
        device = _get_clustering_device()
        # Should produce scipy-equivalent output via the fallback path
        result = _gpu_cdist(a, b, "cosine", device)
        expected = scipy_cdist(a, b, metric="cosine")
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_disable_env_forces_cpu_device(self, monkeypatch) -> None:
        monkeypatch.setenv("PYANNOTE_CLUSTERING_DISABLE_GPU", "1")
        device = _get_clustering_device()
        assert device.type == "cpu"


class TestAgglomerativeClusteringIntegration:
    """End-to-end: AgglomerativeClustering with the GPU path must produce
    the same hard clusters as the stock scipy path for fixed-seed inputs."""

    def _cluster(self, embeddings: np.ndarray, num_clusters: int):
        from pyannote.audio.pipelines.clustering import AgglomerativeClustering

        clustering = AgglomerativeClustering().instantiate(
            {"method": "centroid", "min_cluster_size": 0, "threshold": 0.0}
        )
        return clustering.cluster(
            embeddings=embeddings,
            min_clusters=num_clusters,
            max_clusters=num_clusters,
            num_clusters=num_clusters,
        )

    def test_small_synthetic_parity(self, monkeypatch) -> None:
        """Same synthetic input under GPU vs CPU-forced path yields same clusters."""
        rng = np.random.default_rng(123)
        centers = rng.standard_normal((3, 64)).astype(np.float32)
        labels = np.repeat(np.arange(3), 20)
        embeddings = centers[labels] + 0.05 * rng.standard_normal((60, 64)).astype(np.float32)

        # GPU (or auto-detect) path
        gpu_clusters = self._cluster(embeddings.copy(), num_clusters=3)

        # CPU-forced path
        monkeypatch.setenv("PYANNOTE_CLUSTERING_DISABLE_GPU", "1")
        cpu_clusters = self._cluster(embeddings.copy(), num_clusters=3)

        # Cluster labels may be permuted but the induced partition must match
        def _partition(a):
            return {frozenset(np.where(a == k)[0].tolist()) for k in np.unique(a)}

        assert _partition(gpu_clusters) == _partition(cpu_clusters)
