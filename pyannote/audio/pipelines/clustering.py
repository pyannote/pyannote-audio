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
import warnings
from enum import Enum
from typing import Tuple

import numpy as np
from einops import rearrange
from hmmlearn.hmm import GaussianHMM
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, ParamDict, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from spectralcluster import (
    AutoTune,
    EigenGapType,
    LaplacianType,
    RefinementName,
    RefinementOptions,
    SpectralClusterer,
    SymmetrizeType,
    ThresholdType,
)

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        super().__init__()
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.expects_num_clusters = expects_num_clusters
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

        if self.expects_num_clusters and num_clusters is None:
            raise ValueError("num_clusters must be provided.")

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

    # def constrained_argmax(
    #     self, soft_clusters: np.ndarray, segmentations: SlidingWindowFeature
    # ) -> np.ndarray:
    #     """

    #     Parameters
    #     ----------
    #     soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
    #     segmentations : SlidingWindowFeature
    #         Binarized segmentation.

    #     Returns
    #     -------
    #     hard_clusters : (num_chunks, num_speakers)-shaped array
    #         Hard cluster assignment with

    #     """

    #     import cvxpy as cp

    #     num_chunks, num_speakers, num_clusters = soft_clusters.shape
    #     hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

    #     for c, (scores, (chunk, segmentation)) in enumerate(
    #         zip(soft_clusters, segmentations)
    #     ):

    #         # scores : (num_speakers, num_clusters) array
    #         # segmentation : (num_frames, num_speakers) array

    #         assignment = cp.Variable(shape=(num_speakers, num_clusters), boolean=True)
    #         objective = cp.Maximize(cp.sum(cp.multiply(assignment, scores)))

    #         one_cluster_per_speaker_constraints = [
    #             cp.sum(assignment[i]) == 1 for i in range(num_speakers)
    #         ]

    #         # number of frames where both speakers are active
    #         co_occurrence: np.ndarray = segmentation.T @ segmentation
    #         np.fill_diagonal(co_occurrence, 0)
    #         cannot_link = set(
    #             tuple(sorted(x)) for x in zip(*np.where(co_occurrence > 0))
    #         )
    #         cannot_link_constraints = [
    #             assignment[i] + assignment[j] <= 1 for i, j in cannot_link
    #         ]

    #         problem = cp.Problem(
    #             objective, one_cluster_per_speaker_constraints + cannot_link_constraints
    #         )
    #         problem.solve()

    #         if problem.status == "optimal":
    #             hard_clusters[c] = np.argmax(assignment.value, axis=1)
    #         else:
    #             print(f"{co_occurrence=}")
    #             hard_clusters[c] = np.argmax(scores, axis=1)

    #     return hard_clusters

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
        """Apply HMM clustering

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


class AgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold. Only when `expects_num_clusters` is False.
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

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

        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )

        if num_clusters is None:

            max_threshold: float = (
                dendrogram[-min_clusters, 2]
                if min_clusters < num_embeddings
                else -np.inf
            )
            min_threshold: float = (
                dendrogram[-max_clusters, 2]
                if max_clusters < num_embeddings
                else -np.inf
            )

            threshold = min(max(self.threshold, min_threshold), max_threshold)

        else:

            threshold = (
                dendrogram[-num_clusters, 2]
                if num_clusters < num_embeddings
                else -np.inf
            )

        return fcluster(dendrogram, threshold, criterion="distance") - 1


class OracleClustering(BaseClustering):
    """Oracle clustering"""

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

    def __call__(
        self,
        segmentations: SlidingWindowFeature = None,
        file: AudioFile = None,
        frames: SlidingWindow = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply agglomerative clustering

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


class SpectralClustering(BaseClustering):
    """Spectral clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
        Only "cosine" is currenty supported.
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    laplacian : {"Affinity", "Unnormalized", "RandomWalk", "GraphCut"}
        Laplacian to use.
    eigengap : {"Ratio", "NormalizedDiff"}
        Eigengap approach to use.
    refinement : dict
        Sequence of refinement operations
    refinement["CropDiagonal"] : boolean
    refinement["Symmetrize"] : boolean
    refinement["Diffuse"] : boolean
    refinement["RowWiseNormalized"] : boolean
    symmetrize_type : {"Max", "Average"}
        How to symmetrize the matrix (when refinement["Symmetrize"] is True)
    thresholding_type : {"RowMax", "Percentile"}
        Type of thresholding operation.
    thresholding_with_binarization : boolean
        Set values larger than the threshold to 1.
    thresholding_preserve_diagonal : boolean
        In the row wise thresholding operation, set diagonals of the
        affinity matrix to 0 at the beginning, and back to 1 in the end

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        if metric != "cosine":
            raise ValueError(
                f"Only 'cosine' metric is currently supported in by spectral clustering (you passed {metric=})"
            )

        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )

        # define sequence of refinement operations
        self.refinement = ParamDict(
            CropDiagonal=Categorical([True, False]),
            # gaussian blur does not make sense here because
            # embeddings are not ordered chronologically
            GaussianBlur=False,
            # row-wise threshold is needed by "autotune"
            RowWiseThreshold=True,
            Symmetrize=Categorical([True, False]),
            Diffuse=Categorical([True, False]),
            RowWiseNormalize=Categorical([True, False]),
        )

        self.symmetrize_type = Categorical(["Max", "Average"])

        # for row-wise thresholding
        self.thresholding_type = Categorical(["RowMax", "Percentile"])
        self.thresholding_with_binarization = Categorical([True, False])
        self.thresholding_preserve_diagonal = Categorical([True, False])

        if not self.expects_num_clusters:
            self.eigengap = Categorical(["Ratio", "NormalizedDiff"])

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        **kwargs,
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

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        # see https://github.com/wq2012/SpectralCluster/issues/40
        num_embeddings, _ = embeddings.shape
        if num_embeddings < 20:
            warnings.warn(
                f"Not enough embeddings ({num_embeddings}) to perform reliable spectral clustering."
            )
            return np.zeros((num_embeddings,), dtype=np.int8)

        # Refinements
        refinement_sequence = [
            RefinementName[name] for name, active in self.refinement.items() if active
        ]
        refinement_options = RefinementOptions(
            refinement_sequence=refinement_sequence,
            thresholding_soft_multiplier=0.01,
            thresholding_type=ThresholdType[self.thresholding_type],
            thresholding_with_binarization=self.thresholding_with_binarization,
            thresholding_preserve_diagonal=self.thresholding_preserve_diagonal,
            symmetrize_type=SymmetrizeType[self.symmetrize_type],
        )

        # Laplacian
        laplacian_type = LaplacianType[self.laplacian]

        # Autotune
        if self.expects_num_clusters:
            autotune = None
            eigengap_type = EigenGapType.Ratio
        else:
            autotune = AutoTune(
                p_percentile_min=0.40,
                p_percentile_max=0.95,
                init_search_step=0.05,
                search_level=1,
            )
            eigengap_type = EigenGapType[self.eigengap]

        # Clustering
        clusters = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            refinement_options=refinement_options,
            autotune=autotune,
            laplacian_type=laplacian_type,
            eigengap_type=eigengap_type,
        ).predict(embeddings)

        return clusters


class HiddenMarkovModelClustering(BaseClustering):
    """Hidden Markov Model with Gaussian states"""

    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        if metric not in ["euclidean", "cosine"]:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        super().__init__(
            metric=metric,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        self.covariance_type = Categorical(["spherical", "diag", "full", "tied"])
        if not self.expects_num_clusters:
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

        if min_clusters < 2:
            return np.zeros((num_embeddings,), dtype=np.int8)

        if self.metric == "cosine":
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                euclidean_embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=-1, keepdims=True
                )
        elif self.metric == "euclidean":
            euclidean_embeddings = embeddings

        if num_clusters is None:

            history = [
                -np.inf,
            ]

            patience = min(3, max_clusters - min_clusters)

            num_clusters = min_clusters - 1
            best_criterion = -np.inf

            for n_components in range(min_clusters, max_clusters + 1):

                hmm = self.fit_hmm(n_components, euclidean_embeddings)
                train_clusters = hmm.predict(euclidean_embeddings)

                # compute distance between centroids
                centroids = np.vstack(
                    [
                        np.mean(embeddings[train_clusters == k], axis=0)
                        for k in range(n_components)
                    ]
                )
                centroids_pdist = pdist(centroids, metric=self.metric)
                current_criterion = np.min(centroids_pdist)

                # stop adding states when distance between two closest states
                #  * no longer increases
                #  * no longer goes above {threshold}

                # final number of states is the last one for which the criterion
                # goes above {threshold}.

                # THIS IS A TERRIBLE CRITERION THAT NEEDS TO BE FIXED
                best_criterion = max(history)
                increasing = current_criterion > best_criterion
                big_enough = current_criterion > self.threshold
                if increasing or big_enough:
                    num_clusters = n_components
                    best_criterion = max(best_criterion, current_criterion)
                elif n_components == num_clusters + patience:
                    break

                history.append(current_criterion)

        if num_clusters == 1:
            return np.zeros((num_embeddings,), dtype=np.int8)

        # once num_clusters is estimated, fit the HMM several times
        # and keep the one that best fits the data
        hmm = self.fit_hmm(num_clusters, euclidean_embeddings)

        try:
            train_clusters = hmm.predict(euclidean_embeddings)
        except ValueError:
            # ValueError: startprob_ must sum to 1 (got nan)
            train_clusters = np.zeros((num_embeddings,), dtype=np.int8)

        return train_clusters


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    SpectralClustering = SpectralClustering
    HiddenMarkovModelClustering = HiddenMarkovModelClustering
    OracleClustering = OracleClustering
