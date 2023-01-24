# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
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


import random
from enum import Enum
from typing import Tuple

import numpy as np
from einops import rearrange
from hmmlearn.hmm import GaussianHMM
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import (
    Categorical,
    Integer,
    LogUniform,
    ParamDict,
    Uniform,
)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate

try:
    from finch import FINCH

    FINCH_IS_AVAILABLE = True

except ImportError:
    FINCH_IS_AVAILABLE = False


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):

        super().__init__()
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
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
        segmentations: SlidingWindowFeature = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter NaN embeddings and downsample embeddings

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """

        chunk_idx, speaker_idx = np.where(~np.any(np.isnan(embeddings), axis=2))

        # sample max_num_embeddings embeddings
        num_embeddings = len(chunk_idx)
        if num_embeddings > self.max_num_embeddings:
            indices = list(range(num_embeddings))
            random.shuffle(indices)
            indices = sorted(indices[: self.max_num_embeddings])
            chunk_idx = chunk_idx[indices]
            speaker_idx = speaker_idx[indices]

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
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
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

        # TODO: add a flag to revert argmax for trainign subset
        # hard_clusters[train_chunk_idx, train_speaker_idx] = train_clusters

        return hard_clusters, soft_clusters

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
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
            return hard_clusters, soft_clusters

        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )

        hard_clusters, soft_clusters = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters


class FINCHClustering(BaseClustering):
    """FINCH clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):

        if not FINCH_IS_AVAILABLE:
            raise ImportError(
                "'finch-clust' must be installed to use FINCH clustering. "
                "Visit https://pypi.org/project/finch-clust/ for installation instructions."
            )

        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(["average", "complete", "single"])

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
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
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # apply FINCH clustering and keep (supposedly pure) penultimate partition
        clusters, _, _ = FINCH(
            embeddings,
            initial_rank=None,
            req_clust=None,
            distance=self.metric,
            ensure_early_exit=True,
            verbose=False,
        )

        _, num_partitions = clusters.shape
        if num_partitions < 2:
            clusters = clusters[:, 0]
        else:
            clusters = clusters[:, -2]
        num_clusters = np.max(clusters) + 1

        # compute centroids
        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_clusters)]
        )

        # perform agglomerative clustering on centroids
        dendrogram = linkage(centroids, metric=self.metric, method=self.method)
        klusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1

        # update clusters
        clusters = -clusters
        for i, k in enumerate(klusters):
            clusters[clusters == -i] = k

        # TODO: handle min/max/num_clusters
        # TODO: handle min_cluster_size

        return clusters


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

    Usage
    -----
    >>> clustering = AgglomerativeClustering(metric="cosine")
    >>> clustering.instantiate({"method": "average",
    ...                         "threshold": 1.0,
    ...                         "min_cluster_size": 1})
    >>> clusters, _  = clustering(embeddings,           # shape
    ...                           num_clusters=None,
    ...                           min_clusters=None,
    ...                           max_clusters=None)
    where `embeddings` is a np.ndarray with shape (num_embeddings, embedding_dimension)
    and `clusters` is a np.ndarray with shape (num_embeddings, )
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):

        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
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
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
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
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        # other methods work just fine with any metric
        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
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

        if num_clusters is not None:

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
        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters


class OracleClustering(BaseClustering):
    """Oracle clustering"""

    def __call__(
        self,
        segmentations: SlidingWindowFeature = None,
        file: AudioFile = None,
        frames: SlidingWindow = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply oracle clustering

        Parameters
        ----------
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

        return hard_clusters, soft_clusters


class HiddenMarkovModelClustering(BaseClustering):
    """Hidden Markov Model with Gaussian states"""

    def __init__(
        self,
        metric: str = "cosine",
        constrained_assignment: bool = False,
    ):

        if metric not in ["euclidean", "cosine"]:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        super().__init__(
            metric=metric,
            constrained_assignment=constrained_assignment,
        )

        self.single_cluster_detection = ParamDict(
            quantile=LogUniform(1e-3, 1e-1),
            threshold=Uniform(0.0, 2.0),
        )

        self.covariance_type = Categorical(["spherical", "diag", "full", "tied"])
        self.threshold = Uniform(0.0, 2.0)

    def filter_embeddings(
        self, embeddings: np.ndarray, segmentations: SlidingWindowFeature
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        train_embeddings : (num_steps, dimension) array
        chunk_idx : (num_steps, ) array
        speaker_idx : (num_steps, ) array

        """

        num_chunks, _, _ = embeddings.shape

        # focus on center of each chunk
        duration = segmentations.sliding_window.duration
        step = segmentations.sliding_window.step

        ratio = 0.5 * (duration - step) / duration
        center_segmentations = Inference.trim(segmentations, warm_up=(ratio, ratio))
        #   shape: num_chunks, num_center_frames, num_speakers

        # number of frames during which speakers are active
        # in the center of the chunk
        num_active_frames: np.ndarray = np.sum(center_segmentations.data, axis=1)
        #   shape: (num_chunks, num_speakers)

        priors = num_active_frames / (
            np.sum(num_active_frames, axis=1, keepdims=True) + 1e-8
        )
        #   shape: (num_chunks, local_num_speakers)

        speaker_idx = np.argmax(priors, axis=1)
        # (num_chunks, )

        # TODO: generate alternative sequences that only differs from train_embeddings
        # in regions where there is overlap.

        train_embeddings = embeddings[range(num_chunks), speaker_idx]
        # (num_chunks, dimension)

        # remove chunks with one of the following property:
        # * there is no active speaker in the center of the chunk
        # * embedding extraction has failed for the most active speaker in the center of the chunk
        center_is_non_speech = np.max(num_active_frames, axis=1) == 0.0
        embedding_is_invalid = np.any(np.isnan(train_embeddings), axis=1)
        chunk_idx = np.where(~(embedding_is_invalid | center_is_non_speech))[0]
        # (num_chunks, )

        return (train_embeddings[chunk_idx], chunk_idx, speaker_idx[chunk_idx])

    def fit_hmm(self, n_components, train_embeddings):

        hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=42,
            implementation="log",
            verbose=False,
        )
        hmm.fit(train_embeddings)

        return hmm

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
    ):

        num_embeddings = len(embeddings)

        # FIXME
        if max_clusters == num_embeddings:
            max_clusters = min(max_clusters, 20)

        if self.metric == "cosine":
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                euclidean_embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=-1, keepdims=True
                )
        elif self.metric == "euclidean":
            euclidean_embeddings = embeddings

        # when the number of clusters is provided, fit a HMM with
        # that many states and return the decoded sequence of states
        if num_clusters is not None:
            hmm = self.fit_hmm(num_clusters, euclidean_embeddings)

            try:
                train_clusters = hmm.predict(euclidean_embeddings)
            except ValueError:
                # ValueError: startprob_ must sum to 1 (got nan)
                # TODO: display a warning that something went wrong
                train_clusters = np.zeros((num_embeddings,), dtype=np.int8)

            return train_clusters

        # heuristic for detecting cases where there is just one large cluster
        # (and a few meaningless outliers)
        if min_clusters == 1:

            # Example with quantile = 1% and threshold = 0.4:
            # if 99% (100% - 1%) of pairwise distance are smaller than 0.4,
            # then we assume that the others are outliers and return one cluster
            if (
                np.quantile(
                    pdist(euclidean_embeddings, metric="euclidean"),
                    1.0 - self.single_cluster_detection["quantile"],
                )
                < self.single_cluster_detection["threshold"]
            ):

                return np.zeros((num_embeddings,), dtype=np.int8)

            # otherwise, we make sure to return at least 2 clusters
            min_clusters = max(2, min_clusters)
            max_clusters = max(2, max_clusters)

        # fit a HMM with increasing number of states and stop adding
        # when the distance between the two closest states
        #  - either no longer increases
        #  - or no longer goes above a threshold
        # the selected number of states is the last one for which the
        # criterion goes above {threshold}.

        # THIS IS A TERRIBLE CRITERION THAT NEEDS TO BE FIXED

        history = [-np.inf]
        patience = min(3, max_clusters - min_clusters)
        num_clusters = min_clusters

        for n_components in range(min_clusters, max_clusters + 1):

            hmm = self.fit_hmm(n_components, euclidean_embeddings)
            try:
                train_clusters = hmm.predict(euclidean_embeddings)
            except ValueError:  # ValueError: startprob_ must sum to 1 (got nan)
                # stop adding states as there too many and not enough
                # training data to train it in a reliable manner.
                break

            # stop early if too few states were found
            if len(np.unique(train_clusters)) < n_components:
                break

            # compute distance between the two closest centroids
            centroids = np.vstack(
                [
                    np.mean(embeddings[train_clusters == k], axis=0)
                    for k in range(n_components)
                ]
            )
            centroids_pdist = pdist(centroids, metric=self.metric)
            current_criterion = np.min(centroids_pdist)

            increasing = current_criterion > max(history)
            big_enough = current_criterion > self.threshold

            if increasing or big_enough:
                num_clusters = n_components

            elif n_components == num_clusters + patience:
                break

            history.append(current_criterion)

        hmm = self.fit_hmm(num_clusters, euclidean_embeddings)
        try:
            train_clusters = hmm.predict(euclidean_embeddings)
        except ValueError:
            # ValueError: startprob_ must sum to 1 (got nan)
            train_clusters = np.zeros((num_embeddings,), dtype=np.int8)

        return train_clusters


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    FINCHClustering = FINCHClustering
    HiddenMarkovModelClustering = HiddenMarkovModelClustering
    OracleClustering = OracleClustering
