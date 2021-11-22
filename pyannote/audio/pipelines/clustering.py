# The MIT License (MIT)
#
# Copyright (c) 2021 CNRS
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


from enum import Enum

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
from sklearn.cluster import OPTICS as SKLearnOPTICS
from sklearn.cluster import AffinityPropagation as SKLearnAffinityPropagation

# from sklearn.cluster import AgglomerativeClustering as SKLearnAgglomerativeClustering
from sklearn_extra.cluster import KMedoids as SKKMedoids
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

from pyannote.core.utils.hierarchy import linkage
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform


class AffinityPropagation(Pipeline):
    def __init__(self, expects_num_clusters: bool = False):
        super().__init__()
        if expects_num_clusters:
            raise NotImplementedError(
                "AffinityPropagation clustering algorithm does not support predefined number of clusters."
            )
        self.damping = Uniform(0.5, 1.0)
        self.preference = Uniform(-50.0, 0.0)  # check what this interval should be

    def initialize(self):
        self._clustering = SKLearnAffinityPropagation(
            damping=self.damping,
            max_iter=200,
            convergence_iter=15,
            copy=True,
            preference=self.preference,
            affinity="precomputed",
            verbose=False,
            random_state=1337,  # for reproducibility
        )

    def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:
        if num_clusters is not None:
            raise NotImplementedError(
                "AffinityPropagation clustering algorithm does not support predefined number of clusters."
            )
        return self._clustering.fit_predict(affinity)


class KMedoids(Pipeline):
    def __init__(self, expects_num_clusters: bool = True):
        super().__init__()
        if not expects_num_clusters:
            raise NotImplementedError(
                "KMedoids clustering algorithm expects the number of clusters to be given."
            )

    def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:
        if num_clusters is None:
            raise ValueError(
                "KMedoids clustering algorithm expects the number of clusters (num_clusters) to be given (you did not provide one)."
            )
        clustering = SKKMedoids(
            n_clusters=num_clusters,
            metric="precomputed",
            method="pam",
            init="heuristic",
            max_iter=300,
            random_state=1337,
        )
        return clustering.fit_predict(np.clip(1.0 - affinity, 0.0, 1.0))


class DBSCAN(Pipeline):
    def __init__(self, expects_num_clusters: bool = False):
        super().__init__()
        if expects_num_clusters:
            raise NotImplementedError(
                "DBSCAN clustering algorithm does not support predefined number of clusters."
            )

        self.eps = Uniform(0.0, 1.0)
        self.min_samples = Integer(2, 100)

    def initialize(self):
        self._clustering = SKLearnDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            algorithm="auto",
            leaf_size=30,
            n_jobs=None,
        )

    def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:
        if num_clusters is not None:
            raise NotImplementedError(
                "DBSCAN clustering algorithm does not support predefined number of clusters."
            )
        return self._clustering.fit_predict(np.clip(1.0 - affinity, 0.0, 1.0))


class OPTICS(Pipeline):
    def __init__(self, expects_num_clusters: bool = False):
        super().__init__()
        if expects_num_clusters:
            raise NotImplementedError(
                "OPTICS clustering algorithm does not support predefined number of clusters."
            )

        self.min_samples = Integer(2, 100)
        self.max_eps = Uniform(0.0, 1.0)
        self.xi = Uniform(0.0, 1.0)

    def initialize(self):
        self._clustering = SKLearnOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric="precomputed",
            cluster_method="xi",
            xi=self.xi,
            predecessor_correction=True,
            min_cluster_size=None,
            algorithm="auto",
            leaf_size=30,
            memory=None,
            n_jobs=None,
        )

    def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:
        if num_clusters is not None:
            raise NotImplementedError(
                "OPTICS clustering algorithm does not support predefined number of clusters."
            )
        return self._clustering.fit_predict(np.clip(1.0 - affinity, 0.0, 1.0))


# class AgglomerativeClustering(Pipeline):
#     def __init__(self, expects_num_clusters: bool = False):
#         super().__init__()
#         self.expects_num_clusters = expects_num_clusters
#         self.linkage = Categorical(["complete", "average", "single"])
#         if not self.expects_num_clusters:
#             self.distance_threshold = Uniform(0.0, 1.0)

#     def initialize(self):
#         if not self.expects_num_clusters:
#             self._clustering = SKLearnAgglomerativeClustering(
#                 n_clusters=None,
#                 affinity="precomputed",
#                 linkage=self.linkage,
#                 distance_threshold=self.distance_threshold,
#             )

#     def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:

#         if num_clusters is None:
#             return self._clustering.fit_predict(1.0 - affinity)

#         else:
#             clustering = SKLearnAgglomerativeClustering(
#                 n_clusters=num_clusters, affinity="precomputed", linkage=self.linkage,
#             )
#             return clustering.fit_predict(1.0 - affinity)


class AgglomerativeClustering(Pipeline):
    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

        self.metric = metric

        self.expects_num_clusters = expects_num_clusters
        if not self.expects_num_clusters:
            self.distance_threshold = Uniform(
                0.0, 2.0
            )  # assume unit-normalized embeddings

        self.method = Categorical(
            ["average", "centroid", "complete", "median", "pool", "single", "ward"]
        )

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,
    ) -> np.ndarray:

        if self.expects_num_clusters and num_clusters is None:
            raise ValueError()

        dendrogram = linkage(embeddings, method=self.method, metric=self.metric)

        if num_clusters is None:
            distance_threshold = self.distance_threshold
        else:
            distance_threshold = dendrogram[-num_clusters, 2]

        return fcluster(dendrogram, distance_threshold, criterion="distance") - 1


class SpectralClustering(Pipeline):
    def __init__(self, expects_num_clusters: bool = False):
        super().__init__()
        self.expects_num_clusters = expects_num_clusters
        self.autotune = Categorical([True, False])
        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )

    def initialize(self):

        self._autotune = None
        self._refinement_options = None

        if self.autotune:
            self._autotune = AutoTune(
                p_percentile_min=0.50,
                p_percentile_max=0.95,
                init_search_step=0.01,
                search_level=1,
            )

            self._refinement_options = RefinementOptions(
                thresholding_soft_multiplier=0.01,
                thresholding_type=ThresholdType.Percentile,
                thresholding_with_binarization=True,
                thresholding_preserve_diagonal=True,
                symmetrize_type=SymmetrizeType.Average,
                refinement_sequence=[
                    RefinementName.RowWiseThreshold,
                    RefinementName.Symmetrize,
                ],
            )

        self._laplacian_type = LaplacianType[self.laplacian]

        if not self.expects_num_clusters:
            self._clustering = SpectralClusterer(
                min_clusters=None,
                max_clusters=None,
                refinement_options=self._refinement_options,
                autotune=self._autotune,
                laplacian_type=self._laplacian_type,
                stop_eigenvalue=1e-2,
                row_wise_renorm=False,
                custom_dist="cosine",
                max_iter=300,
                constraint_options=None,
                eigengap_type=EigenGapType.Ratio,
                affinity_function=lambda precomputed: precomputed,  # precomputed affinity
            )

    def __call__(self, affinity: np.ndarray, num_clusters: int = None) -> np.ndarray:

        if num_clusters is None:
            clustering = self._clustering

        else:
            clustering = SpectralClusterer(
                min_clusters=num_clusters,
                max_clusters=num_clusters,
                refinement_options=self._refinement_options,
                autotune=self._autotune,
                laplacian_type=self._laplacian_type,
                stop_eigenvalue=1e-2,
                row_wise_renorm=False,
                custom_dist="cosine",
                max_iter=300,
                constraint_options=None,
                eigengap_type=EigenGapType.Ratio,
                affinity_function=lambda precomputed: precomputed,  # precomputed affinity
            )

        try:
            labels = clustering.predict(affinity)
        except ValueError:
            num_items, _ = affinity.shape
            labels = np.zeros((num_items,), dtype=np.int)

        return labels


class Clustering(Enum):
    AffinityPropagation = AffinityPropagation
    AgglomerativeClustering = AgglomerativeClustering
    DBSCAN = DBSCAN
    KMedoids = KMedoids
    OPTICS = OPTICS
    SpectralClustering = SpectralClustering
