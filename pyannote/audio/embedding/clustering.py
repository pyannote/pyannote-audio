#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import itertools
import numpy as np
import networkx as nx
from sortedcontainers import SortedSet
from sortedcollections import ValueSortedDict

from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import l2_normalize
from scipy.spatial.distance import squareform


class HierarchicalPoolingClustering(object):
    """Embedding clustering

    Parameters
    ----------
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'angular'.
    pooling_func : callable
        Callable that returns one embedding out of multiple embeddings.
        Defaults to "lambda fX: np.mean(fX, axis=0)"."

    Usage
    -----
    >>> clustering = HierarchicalPoolingClustering()
    >>> annotation = clustering.fit(segmentation, embedding).apply(threshold)
    """

    def __init__(self, metric='angular', pooling_func=None):
        super(HierarchicalPoolingClustering, self).__init__()
        self.metric = metric

        if pooling_func is None:
            pooling_func = lambda fX: np.mean(fX, axis=0)
        self.pooling_func = pooling_func

    def fit(self, segmentation, embeddings):
        """Precompute complete dendrogram

        Parameters
        ----------
        segmentation : pyannote.core.Annotation
            Input segmentation.
        embeddings : pyannote.core.SlidingWindowFeature
            Precomputed embeddings.

        Returns
        -------
        clustering : HierarchicalPoolingClustering
        """

        # extract one embedding per label
        fX = self.initialize_(segmentation, embeddings)

        # compute complete dendrogram (all the way up to one cluster)
        self.dendrogram_ = self.cluster_(fX)

        return self

    def apply(self, threshold):
        """Flatten dendrogram at given threshold

        Parameters
        ----------
        threshold : float
            Stopping criterion.

        Returns
        -------
        annotation : pyannote.core.Annotation
            Clustering result.

        """

        # flatten dendrogram at given threshold
        y = self.flatten_(self.dendrogram_, threshold)

        # rename labels to their cluster ID
        mapping = {i: k for i, k in enumerate(y)}
        return self.annotation_.rename_labels(mapping=mapping)

    def initialize_(self, annotation, features):
        """Compute one embedding per label

        Parameters
        ----------
        annotation : pyannote.core.Annotation
            Input segmentation.
        features : pyannote.core.SlidingWindowFeature
            Precomputed embeddings.

        Returns
        -------
        fX : (n_labels, dimension) numpy array
            One embedding per initial label.
        """

        # rename labels to 0, 1, 2, 3, ...
        self.annotation_ = annotation.rename_labels(generator='int')

        # sorted labels
        labels = sorted(self.annotation_.labels(), key=int)

        # one embedding per label
        n = len(labels)
        _, dimension = features.data.shape
        fX = np.zeros((n, dimension))

        for l, label in enumerate(labels):
            fX_ = features.crop(self.annotation_.label_timeline(label),
                                mode='center')
            fX[l] = self.pooling_func(fX_)

        return fX

    def cluster_(self, fX):
        """Compute complete dendrogram

        Parameters
        ----------
        fX : (n_items, dimension) np.array
            Embeddings.

        Returns
        -------
        dendrogram : list of (i, j, distance) tuples
            Dendrogram.
        """

        N = len(fX)

        # clusters contain the identifier of each cluster
        clusters = SortedSet(np.arange(N))

        # labels[i] = c means ith item belongs to cluster c
        labels = np.array(np.arange(N))

        squared = squareform(pdist(fX, metric=self.metric))
        distances = ValueSortedDict()
        for i, j in itertools.combinations(range(N), 2):
            distances[i, j] = squared[i, j]

        dendrogram = []

        for _ in range(N-1):

            # find most similar clusters
            (c_i, c_j), d = distances.peekitem(index=0)

            # keep track of this iteration
            dendrogram.append((c_i, c_j, d))

            # index of clusters in 'clusters' and 'fX'
            i = clusters.index(c_i)
            j = clusters.index(c_j)

            # merge items of cluster c_j into cluster c_i
            labels[labels == c_j] = c_i

            # update c_i representative
            fX[i] += fX[j]

            # remove c_j cluster
            fX[j:-1, :] = fX[j+1:, :]
            fX = fX[:-1]

            # remove distances to c_j cluster
            for c in clusters[:j]:
                distances.pop((c, c_j))
            for c in clusters[j+1:]:
                distances.pop((c_j, c))

            clusters.remove(c_j)

            if len(clusters) < 2:
                continue

            # compute distance to new c_i cluster
            new_d = cdist(fX[i, :].reshape((1, -1)), fX, metric=self.metric).squeeze()
            for c_k, d in zip(clusters, new_d):

                if c_k < c_i:
                    distances[c_k, c_i] = d
                elif c_k > c_i:
                    distances[c_i, c_k] = d

        return dendrogram

    def flatten_(self, dendrogram, threshold):
        """
        Parameters
        ----------
        dendrogram : list of (i, j, distance) tuples
            Dendrogram.
        threshold : float
            Stopping criterion.

        Returns
        -------
        y : (n_items, ) np.array
            Cluster assignments of each item.
        """

        # dendrogram is expected to go all the way down to just one cluster.
        # therefore, we can infer the initial number of items from its length.
        n_items = len(dendrogram) + 1

        # initialize graph with one node per item
        G = nx.Graph()
        G.add_nodes_from(range(n_items))

        # connect items that belong to the same cluster
        # as long as they are less than "threshold" apart from each other
        for c_i, c_j, d in dendrogram:
            if d > threshold:
                break
            G.add_edge(c_i, c_j)

        # clusters are connected components in the result graph
        y = np.zeros((n_items), dtype=np.int8)
        # clusters = nx.connected_components(G, key=len, reverse=True)
        clusters = nx.connected_components(G)
        for k, cluster in enumerate(clusters):
            for c_i in cluster:
                y[c_i] = k

        return y


class HDBSCANClustering(object):

    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean'):
        super(HDBSCANClustering, self).__init__()

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def apply(self, fX):
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            metric='precomputed')
        distance_matrix = squareform(pdist(fX, metric=self.metric))

        # apply clustering
        cluster_labels = clusterer.fit_predict(distance_matrix)

        # cluster embedding
        n_clusters = np.max(cluster_labels) + 1

        if n_clusters < 2:
            return np.zeros(fX.shape[0], dtype=np.int)

        fC = l2_normalize(
            np.vstack([np.sum(fX[cluster_labels == k, :], axis=0)
                       for k in range(n_clusters)]))

        # tag each undefined embedding to closest cluster
        undefined = cluster_labels == -1
        closest_cluster = np.argmin(
            cdist(fC, fX[undefined, :], metric=self.metric), axis=0)
        cluster_labels[undefined] = closest_cluster

        return cluster_labels
