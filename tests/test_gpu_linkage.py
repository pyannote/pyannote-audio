"""GPU Lance-Williams centroid-linkage parity tests against scipy.

Phase 3's prior work moved pdist to GPU but scipy's linkage(method='centroid')
was still CPU, which Phase 3 measurement showed was the real bottleneck on
long multi-speaker diarization (138s of the 334s 4.7h wall time). This
module ports the Lance-Williams centroid merge loop to GPU.

Correctness basis: for Euclidean distance, the centroid-linkage Lance-Williams
formula is algebraically equivalent to recomputing the merged centroid
``C_ij = (n_i C_i + n_j C_j) / n_ij`` and measuring ``||C_ij − C_k||``. These
tests verify that the GPU implementation produces a scipy-equivalent dendrogram
on fixed-seed synthetic inputs.

Equivalent dendrogram ≠ byte-identical: due to fp32 vs fp64 and floating-point
summation order, the exact distance values and the merge order for ties may
differ. What MUST match:
  * Number of clusters at any given threshold
  * Cluster membership (induced partition) at each threshold
  * Total number of merges (N-1)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage

from pyannote.audio.pipelines.clustering import (
    _get_clustering_device,
    _gpu_linkage_centroid,
    _gpu_linkage_fits,
)


def _make_clustered_embeddings(
    n_clusters: int,
    per_cluster: int,
    dim: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with ``n_clusters`` Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 5.0
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    embeddings = (
        centers[labels]
        + noise * rng.standard_normal((n_clusters * per_cluster, dim)).astype(np.float32)
    )
    # Unit-normalize (matches pyannote's pre-linkage normalization).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings.astype(np.float32), labels


def _partition(clusters: np.ndarray) -> set[frozenset[int]]:
    """Partition of sample indices by cluster label."""
    return {
        frozenset(np.where(clusters == k)[0].tolist()) for k in np.unique(clusters)
    }


def _has_gpu() -> bool:
    if torch.cuda.is_available():
        return True
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


class TestBudgetGuard:
    def test_cpu_device_returns_none(self) -> None:
        embeddings = np.random.rand(500, 64).astype(np.float32)
        result = _gpu_linkage_centroid(embeddings, torch.device("cpu"))
        assert result is None, "CPU device should fall back to scipy"

    def test_small_n_returns_none(self) -> None:
        """N < 200 falls back to scipy because the GPU loop overhead swamps the gain."""
        embeddings = np.random.rand(100, 64).astype(np.float32)
        device = _get_clustering_device()
        result = _gpu_linkage_centroid(embeddings, device)
        assert result is None, "Small N should fall back"

    def test_budget_fits_for_medium_n(self) -> None:
        """Typical diarization N=1000 fits in the default 200 MB linkage budget."""
        device = _get_clustering_device()
        if device.type == "cpu":
            pytest.skip("no GPU present")
        assert _gpu_linkage_fits(1000, 256, device)


@pytest.mark.skipif(not _has_gpu(), reason="GPU Lance-Williams requires CUDA or MPS")
class TestCentroidLinkageParity:
    """GPU centroid linkage must produce scipy-equivalent clusterings."""

    @pytest.mark.parametrize(
        "n_clusters,per_cluster,dim,noise,seed",
        [
            (3, 100, 64, 0.2, 11),
            (5, 80, 128, 0.3, 22),
            (8, 50, 256, 0.25, 33),
            (10, 30, 128, 0.15, 44),
        ],
    )
    def test_partition_matches(
        self,
        n_clusters: int,
        per_cluster: int,
        dim: int,
        noise: float,
        seed: int,
    ) -> None:
        embeddings, _true_labels = _make_clustered_embeddings(
            n_clusters, per_cluster, dim, noise, seed
        )
        device = _get_clustering_device()

        gpu_dendrogram = _gpu_linkage_centroid(embeddings, device)
        assert gpu_dendrogram is not None, "GPU path should activate at this N"
        assert gpu_dendrogram.shape == (embeddings.shape[0] - 1, 4)

        cpu_dendrogram = scipy_linkage(
            embeddings, method="centroid", metric="euclidean"
        )
        assert cpu_dendrogram.shape == gpu_dendrogram.shape

        # Cut both dendrograms at `n_clusters` clusters; the induced partition
        # must match (cluster labels may be permuted).
        gpu_clusters = fcluster(gpu_dendrogram, t=n_clusters, criterion="maxclust")
        cpu_clusters = fcluster(cpu_dendrogram, t=n_clusters, criterion="maxclust")

        gpu_part = _partition(gpu_clusters)
        cpu_part = _partition(cpu_clusters)

        assert gpu_part == cpu_part, (
            f"partition mismatch at n_clusters={n_clusters}: "
            f"GPU produced {len(gpu_part)} clusters, CPU {len(cpu_part)}"
        )

    def test_merge_count_is_correct(self) -> None:
        embeddings, _ = _make_clustered_embeddings(4, 50, 128, 0.2, seed=99)
        gpu = _gpu_linkage_centroid(embeddings, _get_clustering_device())
        assert gpu is not None
        assert gpu.shape[0] == embeddings.shape[0] - 1
        # Column 3 is the new-cluster size; must be a positive integer ≤ N.
        new_sizes = gpu[:, 3].astype(np.int64)
        assert (new_sizes >= 2).all()
        assert (new_sizes <= embeddings.shape[0]).all()

    def test_cluster_ids_are_valid(self) -> None:
        embeddings, _ = _make_clustered_embeddings(5, 60, 64, 0.1, seed=123)
        N = embeddings.shape[0]
        gpu = _gpu_linkage_centroid(embeddings, _get_clustering_device())
        assert gpu is not None
        ids = gpu[:, :2].astype(np.int64)
        # Every id is either an original leaf (0..N-1) or an earlier internal
        # node (N..N+step-1). scipy guarantees this; check the GPU impl does too.
        for step in range(N - 1):
            a, b = ids[step]
            assert 0 <= a < N + step
            assert 0 <= b < N + step
            assert a != b

    def test_monotonic_merge_distances(self) -> None:
        """Centroid linkage is NOT guaranteed monotonic (unlike single/complete),
        but in well-separated synthetic data the merge distances should be
        weakly increasing until the blobs start merging. This test only checks
        the distance column is finite and non-negative."""
        embeddings, _ = _make_clustered_embeddings(6, 50, 128, 0.2, seed=55)
        gpu = _gpu_linkage_centroid(embeddings, _get_clustering_device())
        assert gpu is not None
        assert np.isfinite(gpu[:, 2]).all()
        assert (gpu[:, 2] >= 0).all()


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestFallbackOnBudget:
    """Budget guard must trigger scipy fallback without raising."""

    def test_tiny_budget_forces_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv("PYANNOTE_LINKAGE_VRAM_BUDGET_MB", "1")
        embeddings, _ = _make_clustered_embeddings(3, 200, 128, 0.2, seed=77)
        result = _gpu_linkage_centroid(embeddings, _get_clustering_device())
        assert result is None, "1 MB budget must trip the guard"
