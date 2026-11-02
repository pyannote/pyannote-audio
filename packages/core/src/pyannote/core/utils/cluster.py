# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

from collections import Counter

import numpy as np
from scipy.spatial.distance import squareform

from .distance import pdist


def chinese_whispers_clustering(
    X, t, method="distance", metric="euclidean", max_iter=1000, init=None
):
    """Chinese whispers clustering

    Parameters
    ----------
    X : `np.ndarray`
        (n_samples, n_dimensions) feature vectors.
    t :
    method : `str`
        Method use to build neighboring graph. Defaults to 'distance'.
        No other method is available yet.
    metric : `str`
        The distance metric to use. See `pdist` function for a list of valid
        distance metrics. Defaults to 'euclidean'.
    max_iter : `int`
        Maximum number of iterations. Defaults to 1000.
    init : `np.ndarray`
        (n_samples, ) array. Initial cluster number.
        Defauts to each item in its own cluster.

    Returns
    -------
    T : `np.ndarray`
        (n_samples, ) array. T[i] is the cluster number to which
        original observation i belongs.

    Reference
    ---------
    Chris Biemann. "Chinese Whispers - an Efficient Graph Clustering Algorithm
    and its Application to Natural Language Processing Problems". Workshop on
    TextGraphs, at HLT-NAACL 2006.

    """

    # TODO. add support for 'precomputed' metric

    if method == "distance":
        distance = pdist(X, metric=metric)
        neighbors = squareform(distance < t)

        # mark item with no neighbor as their own neighbor
        has_no_neighbor = np.sum(neighbors, axis=1) == 0
        for i in range(len(X)):
            neighbors[i, i] = has_no_neighbor[i]
    elif method == "knn":
        msg = "only 'distance' method is supported for now."
        raise NotImplementedError(msg)
        # neighbors = np.argpartition(squareform(distance), t, axis=1)[:,:t]
    else:
        msg = "only 'distance' method is supported for now."
        raise NotImplementedError(msg)

    if init is None:
        # initialize one cluster per item
        clusters = np.arange(len(X))
    else:
        # or use provided initialization
        clusters = np.array(init).reshape(-1)

    # list of indices used to iterate over all items
    indices = np.arange(len(X))

    for _ in range(max_iter):
        # keep track of current state
        prev_clusters = np.copy(clusters)

        # loop on all items in randomized order
        # TODO: option to set random seed for reproducibility
        np.random.shuffle(indices)
        for i in indices:
            # count number of neighbors in each cluster
            counts = Counter(clusters[neighbors[i]])
            # assign item to most common neighbor cluster
            clusters[i] = counts.most_common(n=1)[0][0]

        # ratio of items that have changed clusters
        changed = np.mean(clusters != prev_clusters)

        # stop early if not much has changed since last iteration
        if changed > 1e-4:
            break

    # relabel clusters between 1 and K
    _, clusters = np.unique(clusters, return_inverse=True)
    return clusters + 1
