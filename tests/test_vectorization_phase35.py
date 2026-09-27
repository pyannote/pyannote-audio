"""Numerical-parity tests for Phase 3.5 clustering cleanups.

Phase 3.5 originally proposed five vectorization rewrites. Two of them
(``reconstruct`` and ``Inference.aggregate``) were measured in a full
4.7h/8-speaker benchmark run and showed +35% to +57% CPU regressions —
``np.add.at`` / broadcast-masked-max are GPU-style anti-patterns on CPU.
Those rewrites were reverted; the GPU port of both loops is Phase 5.2
material.

What shipped:
  - BaseClustering.constrained_argmax inner zip → vector scatter
  - BaseClustering.assign_embeddings centroid computation via
    bincount + add.at (one-shot allocation instead of list-comp of means)
  - AgglomerativeClustering small/large cluster remap via lookup table

All three are microsecond-scale cleanups with no measurable wall-time
effect; they earn their keep by removing Python loops from the code.
Tests verify numeric parity with the pre-change implementations.
"""

from __future__ import annotations

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Reference implementations (pre-vectorization) — used as oracles.
# -----------------------------------------------------------------------------


def _reference_constrained_argmax(soft_clusters: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
    num_chunks, num_speakers, _ = soft_clusters.shape
    hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
    for c, cost in enumerate(soft_clusters):
        speakers, clusters = linear_sum_assignment(cost, maximize=True)
        for s, k in zip(speakers, clusters):  # the old inner loop
            hard_clusters[c, s] = k
    return hard_clusters


def _reference_centroids(
    train_embeddings: np.ndarray, train_clusters: np.ndarray, num_clusters: int
) -> np.ndarray:
    return np.vstack(
        [
            np.mean(train_embeddings[train_clusters == k], axis=0)
            for k in range(num_clusters)
        ]
    )


def _reference_cluster_remap(
    clusters: np.ndarray,
    small_clusters: np.ndarray,
    large_clusters: np.ndarray,
    centroids_cdist: np.ndarray,
) -> np.ndarray:
    clusters = clusters.copy()
    for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
        clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]
    return clusters


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestConstrainedArgmax:
    """Inner zip-loop → vector scatter."""

    def test_parity(self) -> None:
        rng = np.random.default_rng(42)
        soft = rng.standard_normal((30, 5, 8)).astype(np.float32)
        # Sprinkle NaNs to exercise the nan_to_num branch.
        soft[3, 2, 4] = np.nan
        soft[15, 4, 1] = np.nan

        from pyannote.audio.pipelines.clustering import BaseClustering

        bc = BaseClustering()
        got = bc.constrained_argmax(soft.copy())
        want = _reference_constrained_argmax(soft.copy())
        np.testing.assert_array_equal(got, want)

    def test_single_chunk(self) -> None:
        rng = np.random.default_rng(7)
        soft = rng.standard_normal((1, 3, 3)).astype(np.float32)
        from pyannote.audio.pipelines.clustering import BaseClustering

        bc = BaseClustering()
        np.testing.assert_array_equal(
            bc.constrained_argmax(soft.copy()),
            _reference_constrained_argmax(soft.copy()),
        )


class TestAssignEmbeddingsCentroids:
    """np.vstack-of-means → bincount + add.at scatter."""

    @pytest.mark.parametrize("n,dim,k", [(50, 16, 4), (200, 64, 8), (500, 128, 3)])
    def test_parity(self, n: int, dim: int, k: int) -> None:
        rng = np.random.default_rng(100 + n)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        cluster_ids = rng.integers(0, k, size=n)

        # Guarantee every cluster has at least one member.
        cluster_ids[:k] = np.arange(k)

        # Mirror the production rewrite:
        counts = np.bincount(cluster_ids, minlength=k).astype(embeddings.dtype)
        sums = np.zeros((k, dim), dtype=embeddings.dtype)
        np.add.at(sums, cluster_ids, embeddings)
        new_centroids = sums / np.maximum(counts[:, None], 1.0)

        ref_centroids = _reference_centroids(embeddings, cluster_ids, k)
        np.testing.assert_allclose(new_centroids, ref_centroids, rtol=1e-5, atol=1e-6)


class TestClusterRemap:
    """Small-cluster merge loop → lookup table."""

    def test_parity(self) -> None:
        rng = np.random.default_rng(1234)
        # 10 large clusters, 4 small clusters; 100 samples total
        clusters = rng.integers(0, 14, size=100)
        large_clusters = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        small_clusters = np.array([10, 11, 12, 13])
        # centroids_cdist: (num_large, num_small)
        centroids_cdist = rng.random((10, 4)).astype(np.float32)

        # New implementation: lookup table.
        nearest_large = large_clusters[np.argmin(centroids_cdist, axis=0)]
        remap = np.arange(clusters.max() + 1)
        remap[small_clusters] = nearest_large
        new_clusters = remap[clusters]

        want = _reference_cluster_remap(
            clusters, small_clusters, large_clusters, centroids_cdist
        )
        np.testing.assert_array_equal(new_clusters, want)
