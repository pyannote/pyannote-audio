# The MIT License (MIT)
#
# Copyright (c) 2021-2025 CNRS
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Clustering pipelines"""

import logging
import os
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.plda import PLDA
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.vbx import cluster_vbx
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from torch.profiler import record_function

logger = logging.getLogger(__name__)

# =============================================================================
# Phase 3: GPU clustering via torch.cdist (VRAM-budgeted)
# =============================================================================
# scipy.spatial.distance.{cdist, pdist} + scipy.cluster.hierarchy.linkage run
# on CPU only. For long multi-speaker diarization (e.g. 4.7h / 21 speakers
# with ~12000 segments), clustering reaches 41% of end-to-end wall time on
# CUDA — the single largest remaining stage per Phase 1 baseline.
#
# The helpers below replace the O(N²) pairwise-distance step with a GPU
# implementation (torch.cdist / manual cosine via F.normalize) while keeping
# scipy.cluster.hierarchy.linkage's Lance-Williams merge on CPU (cheap per
# step, nontrivial to GPU-port without a heavy dependency). The speedup
# therefore targets the distance-computation portion only, but that portion
# dominates the stage on long files.
#
# Hard invariants (Phase 3 acceptance gate):
#   1. numeric parity with scipy within rtol=1e-5 on fixed-seed synthetic
#   2. DER delta < 0.1 pp absolute on all reference files both devices
#   3. Peak VRAM delta ≤ +50 MB over Phase 2 baseline (1.05 GB ceiling)
#   4. Fallback to scipy when budget exceeded (N ≥ ~4000 for default budget)
#
# Configuration:
#   PYANNOTE_CLUSTERING_DEVICE       override auto-detected device
#                                    (cuda, cuda:0, mps, cpu). Default: auto.
#   PYANNOTE_CLUSTERING_VRAM_BUDGET_MB  byte budget for clustering tensors
#                                    (default: 50, from vram-budget-table.md).
#   PYANNOTE_CLUSTERING_DISABLE_GPU  if set, bypass GPU path entirely.
# =============================================================================


def _get_clustering_device() -> torch.device:
    """Return the device to use for GPU clustering, with env override + fallback."""
    if os.environ.get("PYANNOTE_CLUSTERING_DISABLE_GPU"):
        return torch.device("cpu")
    env = os.environ.get("PYANNOTE_CLUSTERING_DEVICE", "").strip()
    if env:
        try:
            dev = torch.device(env)
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            if dev.type == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                return torch.device("cpu")
            return dev
        except (RuntimeError, ValueError):
            logger.warning(
                "Invalid PYANNOTE_CLUSTERING_DEVICE=%r, falling back to auto-detection",
                env,
            )
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gpu_clustering_budget_bytes(device: torch.device) -> int:
    """Byte budget for clustering-stage GPU tensors.

    On CUDA this is min(env-override, free_vram - 200 MB headroom). On MPS
    we trust the env override (no reliable free-memory API). On CPU
    returns 0 so every call falls back to scipy.
    """
    if device.type == "cpu":
        return 0
    budget_mb = int(os.environ.get("PYANNOTE_CLUSTERING_VRAM_BUDGET_MB", "50"))
    if device.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(device)
            # Keep 200 MB safety margin for the embedding-stage peak that
            # may still be resident in the allocator's reserved pool.
            return min(budget_mb * 1024 * 1024, max(0, free - 200 * 1024 * 1024))
        except (RuntimeError, AssertionError):
            return budget_mb * 1024 * 1024
    return budget_mb * 1024 * 1024


def _gpu_clustering_fits(n: int, dim: int, device: torch.device) -> bool:
    """Predict whether clustering on ``(n × dim)`` fp32 embeddings fits the budget.

    Accounts for two normalized copies plus an N×N pairwise matrix plus a
    small (8 MB) torch-allocator overhead for transient buffers. Upper-bounds
    the condensed-pdist path (which uses half the memory of the full matrix).

    The ``_gpu_clustering_budget_bytes`` helper already applies a 200 MB
    headroom against free VRAM on CUDA, so we do NOT double-count that
    here — otherwise even N=500 would fail on a 50 MB budget.
    """
    if device.type == "cpu" or n <= 0:
        return False
    bytes_needed = (2 * n * dim * 4) + (n * n * 4) + (8 * 1024 * 1024)
    return bytes_needed <= _gpu_clustering_budget_bytes(device)


def _gpu_cdist(
    a: np.ndarray, b: np.ndarray, metric: str, device: torch.device
) -> np.ndarray:
    """GPU pairwise distance; mirrors ``scipy.spatial.distance.cdist`` semantics.

    Falls back to scipy when the output would exceed the VRAM budget or for
    metrics without a fast GPU path. Supported fast-path metrics: ``cosine``,
    ``euclidean``.
    """
    n_a, dim = a.shape
    n_b = b.shape[0]
    if not _gpu_clustering_fits(max(n_a, n_b), dim, device):
        logger.debug(
            "GPU clustering fallback (cdist n_a=%d n_b=%d dim=%d device=%s): "
            "budget exceeded",
            n_a, n_b, dim, device.type,
        )
        return cdist(a, b, metric=metric)
    if metric not in ("cosine", "euclidean"):
        return cdist(a, b, metric=metric)
    a_t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
        device, non_blocking=True
    )
    b_t = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(
        device, non_blocking=True
    )
    with torch.inference_mode():
        if metric == "cosine":
            a_n = F.normalize(a_t, dim=1)
            b_n = F.normalize(b_t, dim=1)
            result = 1.0 - a_n @ b_n.T
        else:  # euclidean
            result = torch.cdist(a_t, b_t, p=2)
    out = result.cpu().numpy().astype(a.dtype, copy=False)
    del a_t, b_t, result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _gpu_pdist_condensed(
    a: np.ndarray, metric: str, device: torch.device
) -> np.ndarray:
    """GPU condensed pdist (scipy-ordered upper-triangular, flattened).

    Equivalent to ``scipy.spatial.distance.pdist(a, metric=metric)``.
    Falls back to scipy when the output exceeds the VRAM budget or for
    unsupported metrics.
    """
    n, dim = a.shape
    if not _gpu_clustering_fits(n, dim, device):
        logger.debug(
            "GPU clustering fallback (pdist n=%d dim=%d device=%s): budget exceeded",
            n, dim, device.type,
        )
        return pdist(a, metric=metric)
    if metric not in ("cosine", "euclidean"):
        return pdist(a, metric=metric)
    a_t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
        device, non_blocking=True
    )
    with torch.inference_mode():
        if metric == "cosine":
            a_n = F.normalize(a_t, dim=1)
            full = 1.0 - a_n @ a_n.T
        else:  # euclidean
            full = torch.cdist(a_t, a_t, p=2)
        triu_idx = torch.triu_indices(n, n, offset=1, device=device)
        condensed = full[triu_idx[0], triu_idx[1]]
    out = condensed.cpu().numpy().astype(a.dtype, copy=False)
    del a_t, full, triu_idx, condensed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        constrained_assignment: bool = False,
    ):
        super().__init__()
        self.metric = metric
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int | None = None,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
    ):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature | None = None,
        min_active_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter embeddings before clustering

        Embeddings that are removed:
        * NaN embeddings
        * embeddings speaking less than `min_active_ratio` times the chunk duration

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        min_active_ratio : float, optional
            Minimum active ratio for a speaker to be considered active
            during clustering.

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """

        _, num_frames, _ = segmentations.data.shape

        # frames where only one speaker is active
        single_active_mask = (np.sum(segmentations.data, axis=2, keepdims=True) == 1)
        
        # count number of clean frames per chunk and speaker
        num_clean_frames = np.sum(segmentations.data * single_active_mask, axis=1)
        # num_chunks, num_speakers
                
        # whether speaker is active enough on their own
        active = num_clean_frames >= min_active_ratio * num_frames
        # num_chunks, num_speakers

        # whether speaker embedding extraction went fine
        valid = ~np.any(np.isnan(embeddings), axis=2)

        # indices of embeddings that are both active enough and valid
        chunk_idx, speaker_idx = np.where(active * valid)

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
        
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # compute distance between embeddings and clusters
        with record_function("pyannote::clustering_cdist"):
            e2k_distance = rearrange(
                _gpu_cdist(
                    rearrange(embeddings, "c s d -> (c s) d"),
                    centroids,
                    self.metric,
                    _get_clustering_device(),
                ),
                "(c s) k -> c s k",
                c=num_chunks,
                s=num_speakers,
            )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature | None = None,
        num_clusters: int | None = None,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """

        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        num_embeddings, _ = train_embeddings.shape

        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            # do NOT apply clustering when min_clusters = max_clusters = 1
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        train_clusters = self.cluster(
            train_embeddings,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            num_clusters=num_clusters,
        )

        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters, centroids


class AgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold.
    min_cluster_size : int in range [1, 20]
        Minimum cluster size
    """

    expects_num_clusters: bool = False

    def __init__(
        self,
        metric: str = "cosine",
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
        num_clusters: int | None = None,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        num_embeddings, _ = embeddings.shape

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )

        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            with record_function("pyannote::clustering_normalize"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            with record_function("pyannote::clustering_linkage"):
                # Pre-compute the condensed pairwise distance on GPU, then
                # hand the 1-D distance vector to scipy's Lance-Williams merge.
                # linkage() ignores the ``metric`` kwarg when given a
                # condensed vector.
                dendrogram: np.ndarray = linkage(
                    _gpu_pdist_condensed(
                        embeddings, "euclidean", _get_clustering_device()
                    ),
                    method=self.method,
                )

        # other methods work just fine with any metric
        else:
            with record_function("pyannote::clustering_linkage"):
                dendrogram: np.ndarray = linkage(
                    _gpu_pdist_condensed(
                        embeddings, self.metric, _get_clustering_device()
                    ),
                    method=self.method,
                )

        # apply the predefined threshold
        clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1

        # split clusters into two categories based on their number of items:
        # large clusters vs. small clusters
        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        # force num_clusters to min_clusters in case the actual number is too small
        if num_large_clusters < min_clusters:
            num_clusters = min_clusters

        # force num_clusters to max_clusters in case the actual number is too large
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        # look for perfect candidate if necessary
        if num_clusters is not None and num_large_clusters != num_clusters:
            # switch stopping criterion from "inter-cluster distance" stopping to "iteration index"
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            # traverse the dendrogram by going further and further away
            # from the "optimal" threshold

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.threshold)):
                # only consider iterations that might have resulted
                # in changing the number of (large) clusters
                new_cluster_size = _dendrogram[iteration, 3]
                if new_cluster_size < min_cluster_size:
                    continue

                # estimate number of large clusters at considered iteration
                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)

                # keep track of iteration that leads to the number of large clusters
                # as close as possible to the target number of clusters.
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                # stop traversing the dendrogram as soon as we found a good candidate
                if num_large_clusters == num_clusters:
                    break

            # re-apply best iteration in case we did not find a perfect candidate
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        # re-assign each small cluster to the most similar large cluster based on their respective centroids
        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        with record_function("pyannote::clustering_cdist_merge"):
            centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters


class KMeansClustering(BaseClustering):
    """KMeans clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean"}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    None
    """

    expects_num_clusters: bool = True

    def __init__(
        self,
        metric: str = "cosine",
    ):
        if metric not in ["cosine", "euclidean"]:
            raise ValueError(
                f"Unsupported metric: {metric}. Must be 'cosine' or 'euclidean'."
            )

        super().__init__(metric=metric)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
        num_clusters: int | None = None,
    ):
        """Perform KMeans clustering

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        num_clusters : int, optional
            Expected number of clusters.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        if num_clusters is None:
            raise ValueError("`num_clusters` must be provided.")

        num_embeddings, _ = embeddings.shape
        if num_embeddings < num_clusters:
            # one cluster per embedding as int
            return np.arange(num_embeddings, dtype=np.int32)

        # unit-normalize embeddings to use 'euclidean' distance
        if self.metric == "cosine":
            with record_function("pyannote::clustering_normalize"):
                with np.errstate(divide="ignore", invalid="ignore"):
                    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # perform Kmeans clustering
        with record_function("pyannote::clustering_kmeans"):
            return KMeans(
                n_clusters=num_clusters, n_init=3, random_state=42, copy_x=False
            ).fit_predict(embeddings)


class VBxClustering(BaseClustering):

    expects_num_clusters: bool = False


    def __init__(
        self,
        plda: PLDA,
        metric: str = "cosine",
        constrained_assignment: bool = True,
    ):
        super().__init__(
            metric=metric,
            constrained_assignment=constrained_assignment,
        )

        self.plda = plda

        self.threshold = Uniform(0.5, 0.8)  # assume unit-normalized embeddings
        self.Fa = Uniform(0.01, 0.5)
        self.Fb = Uniform(0.01, 15.0)

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature | None = None,
        num_clusters: int | None = None,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        
        constrained_assignment = self.constrained_assignment

        train_embeddings, _, _ = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )

        if train_embeddings.shape[0] < 2:
            # do NOT apply clustering when the number of training embeddings is less than 2
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        # AHC
        with record_function("pyannote::clustering_normalize"):
            train_embeddings_normed = train_embeddings / np.linalg.norm(
                train_embeddings, axis=1, keepdims=True
            )
        with record_function("pyannote::clustering_linkage"):
            dendrogram = linkage(
                _gpu_pdist_condensed(
                    train_embeddings_normed, "euclidean", _get_clustering_device()
                ),
                method="centroid",
            )
        ahc_clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)

        # VBx

        fea = self.plda(train_embeddings)
        q, sp = cluster_vbx(
            ahc_clusters,
            fea,
            self.plda.phi,
            Fa=self.Fa,
            Fb=self.Fb,
            maxIters=20,
        )

        num_chunks, num_speakers, dimension = embeddings.shape
        W = q[:, sp > 1e-7] # responsibilities of speakers that VBx kept
        centroids = W.T @ train_embeddings.reshape(-1, dimension) / W.sum(0, keepdims=True).T

        # (optional) K-Means
        # re-cluster with Kmeans only in case the automatically determined
        # number of clusters does not match the requested number of speakers
        # (either too low, or too high, or different from the requested number)
        auto_num_clusters, _ = centroids.shape
        if auto_num_clusters < min_clusters:
            num_clusters = min_clusters
        elif auto_num_clusters > max_clusters:
            num_clusters = max_clusters
        if num_clusters and num_clusters != auto_num_clusters:
            # disable constrained assignment when forcing number of clusters
            # as it might results in artificially increasing the number of clusters
            constrained_assignment = False
            kmeans_clusters = KMeans(
                n_clusters=num_clusters, n_init=3, random_state=42, copy_x=False
            ).fit_predict(train_embeddings_normed)
            centroids = np.vstack(
                [
                    np.mean(train_embeddings[kmeans_clusters == k], axis=0)
                    for k in range(num_clusters)
                ])

        # calculate distance
        with record_function("pyannote::clustering_cdist"):
            e2k_distance = rearrange(
                _gpu_cdist(
                    rearrange(embeddings, "c s d -> (c s) d"),
                    centroids,
                    self.metric,
                    _get_clustering_device(),
                ),
                "(c s) k -> c s k",
                c=num_chunks,
                s=num_speakers,
            )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained_assignment:
            const = soft_clusters.min() - 1.   # const < any_valid_score
            soft_clusters[segmentations.data.sum(1) == 0] = const
            hard_clusters = self.constrained_argmax(
                soft_clusters,
            )
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)
        
        return hard_clusters, soft_clusters, centroids


class OracleClustering(BaseClustering):
    """Oracle clustering"""

    expects_num_clusters: bool = True

    def __call__(
        self,
        embeddings: np.ndarray | None = None,
        segmentations: SlidingWindowFeature | None = None,
        file: AudioFile | None = None,
        frames: SlidingWindow | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply oracle clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array, optional
            Sequence of embeddings. When provided, compute speaker centroids
            based on these embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        file : AudioFile
        frames : SlidingWindow

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension), optional
            Clusters centroids if `embeddings` is provided, None otherwise.
        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape
        window = segmentations.sliding_window

        oracle_segmentations = oracle_segmentation(file, window, frames=frames)
        #   shape: (num_chunks, num_frames, true_num_speakers)

        file["oracle_segmentations"] = oracle_segmentations

        _, oracle_num_frames, num_clusters = oracle_segmentations.data.shape

        segmentations = segmentations.data[:, : min(num_frames, oracle_num_frames)]
        oracle_segmentations = oracle_segmentations.data[
            :, : min(num_frames, oracle_num_frames)
        ]

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        soft_clusters = np.zeros((num_chunks, num_speakers, num_clusters))
        for c, (segmentation, oracle) in enumerate(
            zip(segmentations, oracle_segmentations)
        ):
            _, (permutation, *_) = permutate(oracle[np.newaxis], segmentation)
            for j, i in enumerate(permutation):
                if i is None:
                    continue
                hard_clusters[c, i] = j
                soft_clusters[c, i, j] = 1.0

        if embeddings is None:
            return hard_clusters, soft_clusters, None

        (
            train_embeddings,
            train_chunk_idx,
            train_speaker_idx,
        ) = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        train_clusters = hard_clusters[train_chunk_idx, train_speaker_idx]
        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        return hard_clusters, soft_clusters, centroids


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    KMeansClustering = KMeansClustering
    VBxClustering = VBxClustering
    OracleClustering = OracleClustering
