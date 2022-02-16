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


from typing import Optional
from enum import Enum

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from spectralcluster import EigenGapType, LaplacianType, SpectralClusterer

from pyannote.core.utils.distance import pdist
from pyannote.core.utils.hierarchy import linkage
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Uniform


class ClusteringMixin:
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


class AgglomerativeClustering(ClusteringMixin, Pipeline):
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
    method : {"average", "centroid", "complete", "median", "pool", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold. Only when `expects_num_clusters` is False.

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

        self.metric = metric

        self.expects_num_clusters = expects_num_clusters
        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings

        self.method = Categorical(
            ["average", "centroid", "complete", "median", "pool", "single", "ward"]
        )

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ) -> np.ndarray:
        """Apply agglomerative clustering

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) np.ndarray
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Defaults to 1.
            Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Defaults to `num_embeddings`.
            Has no effect when `num_clusters` is provided.

        Returns
        -------
        clusters : (num_embeddings, ) np.ndarray
        """

        num_embeddings, _ = embeddings.shape
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

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


class SpectralClustering(ClusteringMixin, Pipeline):
    """Spectral clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    laplacian : {"Affinity", "Unnormalized", "RandomWalk", "GraphCut"}
        Laplacian to use.
    eigengap : {"Ratio", "NormalizedDiff"}
        Eigengap approach to use.

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()
        self.metric = metric
        self.expects_num_clusters = expects_num_clusters

        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )
        self.eigengap = Categorical(["Ratio", "NormalizedDiff"])

    def _affinity_function(self, embeddings: np.ndarray) -> np.ndarray:
        return squareform(1.0 - 0.5 * pdist(embeddings, metric=self.metric))

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ) -> np.ndarray:
        """Apply spectral clustering

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) np.ndarray
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Defaults to 1.
            Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Defaults to `num_embeddings`.
            Has no effect when `num_clusters` is provided.

        Returns
        -------
        clusters : (num_embeddings, ) np.ndarray
        """

        num_embeddings, _ = embeddings.shape
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        return SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            laplacian_type=LaplacianType[self.laplacian],
            eigengap_type=EigenGapType[self.eigengap],
            affinity_function=self._affinity_function,
        ).predict(embeddings)


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    SpectralClustering = SpectralClustering


class NearestClusterAssignment:
    """
    
    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    allow_reassignment : bool, optional
        Allow already assigned embeddings to be reassigned to a new cluster
        in case it is closer than the original one. Defaults to stick with
        the original cluster.
    
    """

    def __init__(self, metric: str = "cosine", allow_reassignment: bool = False):
        super().__init__()
        self.metric = metric
        self.allow_reassignment = allow_reassignment

    def __call__(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        cannot_link: Optional[np.ndarray] = None,
    ):
        """
        
        Parameters
        ----------
        embeddings : (num_embeddings, dimension) np.ndarray
            Speaker embeddings. NaN embeddings indicate that cluster prior 
            probability must be used to assign them.
        clusters : (num_embeddings, ) np.ndarray
            Clustering output, where 
            * clusters[e] == k  means eth embedding has already been assigned to kth cluster. 
            * clusters[e] == -1 means eth embedding as yet to be assigned.
        cannot_link : (num_embeddings, num_embeddings) np.ndarray
            "cannot link" constraints.  

        Returns
        -------
        new_clusters : (num_embeddings, ) np.ndarray

        """

        num_embeddings = embeddings.shape[0]
        if cannot_link is None:
            cannot_link = np.zeros((num_embeddings, num_embeddings))

        # compute embedding-to-embedding distances
        e2e_distance = squareform(pdist(embeddings, metric=self.metric))

        max_distance = np.nanmax(e2e_distance)
        e2e_distance -= max_distance

        # compute embedding-to-cluster distances
        num_clusters = np.max(clusters) + 1
        e2c_distance = np.vstack(
            [np.mean(e2e_distance[clusters == k], axis=0) for k in range(num_clusters)]
        ).T

        # when embeddings cannot be extracted, arbitrarily use prior probability
        # as distance to cluster
        _, count = np.unique(clusters[clusters != -1], return_counts=True)
        prior = count / np.sum(count)
        e2c_distance[np.any(np.isnan(embeddings), axis=1)] = max_distance + 1.0 - prior

        # without cannot link constraints
        if np.count_nonzero(cannot_link) == 0:
            # assign embeddings to nearest cluster
            new_clusters = np.argmin(e2c_distance, axis=1)

        # with cannot link constraints
        else:
            new_clusters = np.full_like(clusters, -1)
            for _ in range(num_embeddings):
                e, c = np.unravel_index(np.argmin(e2c_distance), e2c_distance.shape)
                new_clusters[e] = c
                e2c_distance[e, :] = np.inf
                e2c_distance[np.nonzero(cannot_link[e]), c] += 1.0

        if self.allow_reassignment:
            return new_clusters
        else:
            return np.where(clusters == -1, new_clusters, clusters)

