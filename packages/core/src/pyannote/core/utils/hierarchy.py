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

"""
==============================================================
Hierarchical clustering (:mod:`pyannote.core.utils.hierarchy`)
==============================================================

"""

from collections import Counter
from collections.abc import Callable
from inspect import signature

import numpy as np
import scipy.cluster.hierarchy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from .distance import cdist, l2_normalize, pdist, to_condensed, to_squared


def linkage(X, method="single", metric="euclidean", **kwargs):
    """Same as scipy.cluster.hierarchy.linkage with more metrics and methods"""

    if method == "pool":
        return pool(X, metric=metric, **kwargs)

    # corner case when using non-euclidean distances with methods
    # designed for the euclidean distance
    if metric != "euclidean" and method in ["centroid", "median", "ward"]:
        # Those 3 methods only work with 'euclidean' distance.
        # Therefore, one has to unit-normalized embeddings before
        # comparison in case they were optimized for 'cosine' (or 'angular')
        # distance.
        X = l2_normalize(X)
        metric = "euclidean"

    distance = pdist(X, metric=metric)
    return scipy.cluster.hierarchy.linkage(
        distance, method=method, metric=metric, **kwargs
    )


def _average_pooling_func(
    u: int,
    v: int,
    S: np.ndarray = None,
    C: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Compute average of newly merged cluster

    Parameters
    ----------
    u : int
        Cluster index.
    v : int
        Cluster index.
    C : (2 x n_observations - 1, dimension) np.ndarray
        Cluster average.
    S : (2 x n_observations - 1, ) np.ndarray
        Cluster size.

    Returns
    -------
    Cuv : (dimension, ) np.ndarray
        Average of newly formed cluster.
    """
    return (C[u] * S[u] + C[v] * S[v]) / (S[u] + S[v])


def _centroid_pooling_func(
    u: int,
    v: int,
    X: np.ndarray = None,
    d: np.ndarray = None,
    K: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Compute centroid of newly merged cluster

    Parameters
    ----------
    u : int
        Cluster index.
    v : int
        Cluster index.
    X : (n_observations, dimension) np.ndarray
        Observations.
    d : (n_observations, n_obversations) np.ndarray
        Distance between observations.
    K : (n_observations, ) np.ndarray, optional
        Cluster assignment.

    Returns
    -------
    Cuv : (dimension, ) np.ndarray
        Centroid of newly formed cluster.
    """
    u_or_v = np.where(np.isin(K, [u, v]))[0]
    i = np.argmin(np.mean(d[u_or_v][:, u_or_v], axis=0))
    return X[u_or_v[i]]


def propagate_constraints(
    cannot_link: list[tuple[int, int]], must_link: list[tuple[int, int]]
):

    # expand list of "cannot link" constraints thanks to the following rule
    # (u != v) & (v == w) ==> u != w

    cannot_link = {tuple(sorted(uv)) for uv in cannot_link}
    while True:
        new_cannot_link = []
        for x, y in must_link:
            for u, v in cannot_link:
                ij = tuple(sorted({u, v}.symmetric_difference({x, y})))
                if not ij:
                    msg = (
                        f"Found a conflict between 'must_link' and "
                        f"'cannot_link' constraints for pair ({u}, {v})."
                    )
                    raise ValueError(msg)
                if len(ij) == 2 and ij not in cannot_link:
                    new_cannot_link.append(ij)
        if new_cannot_link:
            cannot_link.update(new_cannot_link)
        else:
            break

    return list(cannot_link)


def pool(
    X: np.ndarray,
    metric: str = "euclidean",
    pooling_func: str | Callable = "average",
    cannot_link: list[tuple[int, int]] | None = None,
    must_link: list[tuple[int, int]] | None = None,
    must_link_method: str = "both",
):
    """'pool' linkage

    Parameters
    ----------
    X : np.ndarray
        (n_samples, dimension) obversations.
    metric : {"euclidean", "cosine", "angular"}, optional
        Distance metric. Defaults to "euclidean"
    pooling_func: callable, optional
        Defaults to "average".
    cannot_link : list of pairs, optional
        Pairs of indices of observations that cannot be linked. For instance,
        [(1, 2), (5, 6)] means that first and second observations cannot end up
        in the same cluster, as well as 5th and 6th obversations.
    must_link : list of pairs, optional
        Pairs of indices of observations that must be linked. For instance,
        [(1, 2), (5, 6)] means that first and second observations must end up
        in the same cluster, as well as 5th and 6th obversations.
    must_link_method : {"merge", "propagate", "both"}, optional
        Method used for taking "must link" constraints into account.
        * use "merge" to initialize clusters by merging "must link" observations
          before any other regular clustering iterations.
        * use "propagate" to infer additional "cannot link" constraints by
          applying the following propagation rule:
                if u and v cannot be linked and v and w must be linked,
                then u and w cannot be linked.
        * use "both" to apply both methods.
        Defaults to "both".
    """

    if pooling_func == "average":
        pooling_func = _average_pooling_func

    elif pooling_func == "centroid":
        pooling_func = _centroid_pooling_func

    elif isinstance(pooling_func, str):
        msg = (
            f"{pooling_func} pooling is not supported. Choose between "
            f"'average' and 'centroid', or provide your own function."
        )
        raise ValueError(msg)

    if cannot_link is None:
        cannot_link = []

    if must_link is None:
        must_link = []

    # obtain number of original observations
    n, dimension = X.shape

    # compute similarity matrix
    d = pdist(X, metric=metric)

    # K[j] contains the index of the cluster to which
    # the jth observation is currently assigned
    K = np.arange(n)

    # S[k] contains the current size of kth cluster
    S = np.zeros(2 * n - 1, dtype=np.int16)
    S[:n] = 1

    # C[k] contains the centroid of kth cluster
    C = np.zeros((2 * n - 1, dimension))
    # at the beginning, each observation is assigned to its own cluster
    C[:n, :] = X

    # clustering tree (aka dendrogram)
    # Z[i, 0] and Z[i, 1] are merged at ith iteration
    # Z[i, 2] is the distance between Z[i, 0] and Z[i, 1]
    # Z[i, 3] is the total number of original observation in the newly formed cluster
    Z = np.zeros((n - 1, 4))

    # convert condensed pdist matrix for the `n` original observation to a
    # condensed pdist matrix for the `2n-1` clusters (including the `n`
    # original observations) that will exist at some point during the process.
    D = np.inf * np.ones((2 * n - 1) * (2 * n - 2) // 2)
    D[to_condensed(2 * n - 1, *to_squared(n, np.arange(n * (n - 1) // 2)))] = d

    if "d" in signature(pooling_func).parameters:
        d = squareform(d)

    def merge(u, v, iteration, constraint=False):
        """Merge two clusters

        Parameters
        ----------
        u, v : int
            Indices of clusters to merge.
        iteration : int
            Current clustering iteration.
        constraint : bool
            Set to True to indicate that this merge is coming from a 'must_link'
            constraint. This will artificially set Z[iteration, 2] to 0.0.

        Returns
        -------
        uv : int
            Indices of resulting cluster.

        Raises
        ------
        "ValueError" in case of conflict between "must_link" and "cannot_link"
        constraints.

        """

        k = to_condensed(2 * n - 1, u, v)

        if constraint and D[k] == np.inf:
            w = u if u < n else v
            msg = f"Found a conflict between 'must_link' and 'cannot_link' constraints for observation {w}."
            raise ValueError(msg)

        # keep track of ...
        # ... which clusters are merged at this iteration
        # Place the larger child first; preserve input order for equal sizes.
        Z[iteration, 0] = v if S[v] > S[u] else u
        Z[iteration, 1] = u if Z[iteration, 0] == v else v

        # ... their distance
        Z[iteration, 2] = 0.0 if constraint else D[k]

        # ... the size of the newly formed cluster
        Z[iteration, 3] = S[u] + S[v]
        S[n + iteration] = S[u] + S[v]

        # compute "representation" of newly formed cluster
        C[n + iteration] = pooling_func(u, v, X=X, d=d, K=K, S=S, C=C)

        # merged clusters are now empty...
        S[u] = 0
        S[v] = 0

        # move observations of merged clusters into the newly formed cluster
        K[K == u] = n + iteration
        K[K == v] = n + iteration

        # compute distance to newly formed cluster
        # (only for clusters that still exists, i.e. those that are not empty)
        empty = S[: n + iteration] == 0
        k = to_condensed(2 * n - 1, n + iteration, np.arange(n + iteration)[~empty])
        D[k] = cdist(
            C[np.newaxis, n + iteration, :],
            C[: n + iteration, :][~empty, :],
            metric=metric,
        )

        # condensed indices of all (u, _) and (v, _) pairs
        _u = to_condensed(2 * n - 1, u, np.arange(u))
        u_ = to_condensed(2 * n - 1, u, np.arange(u + 1, n + iteration))
        _v = to_condensed(2 * n - 1, v, np.arange(v))
        v_ = to_condensed(2 * n - 1, v, np.arange(v + 1, n + iteration))

        # propagate "cannot link" constraints to newly formed cluster
        if cannot_link:
            x, _ = to_squared(2 * n - 1, _u[D[_u] == np.inf])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.inf
            _, x = to_squared(2 * n - 1, u_[D[u_] == np.inf])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.inf
            x, _ = to_squared(2 * n - 1, _v[D[_v] == np.inf])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.inf
            _, x = to_squared(2 * n - 1, v_[D[v_] == np.inf])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.inf

        # distance to merged clusters u and v no longer exist
        D[_u] = np.inf
        D[u_] = np.inf
        D[_v] = np.inf
        D[v_] = np.inf

        k = to_condensed(2 * n - 1, n + iteration, np.arange(n + iteration)[empty])
        D[k] = np.inf

        return n + iteration

    iteration = 0

    if cannot_link:
        if must_link_method in ["propagate", "both"]:
            cannot_link = propagate_constraints(cannot_link, must_link)

        # take "cannot link" constraints into account by artifically setting the
        # distance between corresponding observations to infinity.
        u, v = zip(*cannot_link)
        D[to_condensed(2 * n - 1, u, v)] = np.inf

    # take "must link" constraints into account by merging corresponding
    # observations regardless of their actual similarity. this might lead to
    # weird clustering results when merged observations are very dissimilar.
    if must_link_method in ["merge", "both"] and must_link:
        # find connected components in "must link" graph
        graph = np.zeros((n, n), dtype=np.int8)
        for u, v in must_link:
            graph[u, v] = 1
        _, K_init = connected_components(
            csr_matrix(graph), directed=False, return_labels=True
        )

        # merge observations within each connected components
        for k, count in Counter(K_init).items():
            if count < 2:
                continue
            u, *others = np.where(K_init == k)[0]
            for v in others:
                u = merge(u, v, iteration, constraint=True)
                iteration += 1

    # iterate until one cluster remains
    start_iteration = iteration
    for iteration in range(start_iteration, n - 1):
        # find two most similar clusters
        k = np.argmin(D)

        # when cannot_link constraints prevents any further merging,
        # choose an arbitrary (u, v) pair among remaining clusters
        if D[k] == np.inf:
            u, v, *_ = np.where(S > 0)[0]
        else:
            u, v = to_squared(2 * n - 1, k)

        _ = merge(u, v, iteration)

    # return dendrogram
    return Z


def fcluster_auto(X, Z, metric="euclidean"):
    """Forms flat clusters using within-class sum of square elbow criterion

    Parameters
    ----------
    X : `np.ndarray`
        (n_samples, n_dimensions) feature vectors.
    Z : `np.ndarray`
        The hierarchical clustering encoded with the matrix returned by the
        `linkage` function.
    metric : `str`
        The distance metric to use. See `pdist` function for a list of valid
        distance metrics.

    Returns
    -------
    T : ndarray
        An array of length n. T[i] is the flat cluster number to which
        original observation i belongs.

    Reference
    ---------
    H. Delgado, X. Anguerra, C. Fredouille, J. Serrano. "Fast Single- and
    Cross-Show Speaker Diarization Using Binary Key Speaker Modeling".
    IEEE Transactions on Audio Speech and Language Processing
    """

    # within-class sum of squares
    wcss = []
    for threshold in Z[:, 2]:
        y_t = scipy.cluster.hierarchy.fcluster(Z, threshold, criterion="distance")
        D = []
        for k in np.unique(y_t):
            Xk = X[y_t == k]
            Ck = np.mean(Xk, axis=0, keepdims=True)
            D.append(
                cdist(Ck, Xk, metric=metric).reshape(
                    -1,
                )
            )
        wcss.append(np.mean(np.hstack(D) ** 2))
    wcss = np.array(wcss)

    # elbow criterion
    n = len(X)

    # after first step, there n-1 clusters (x1)
    x1, y1 = n - 1, wcss[0]
    # after last step, there is 1 cluster (x2)
    x2, y2 = 1, wcss[-1]

    # equation of line passing at both points
    # ax + by + c = 0
    a = (y2 - y1) / (x2 - x1)
    b = 1
    c = (x2 * y1 - x1 * y2) / (x1 - x2)

    # elbow is at maximum distance to this line
    distance = np.abs(a * np.arange(1, n) + b * wcss + c) / np.sqrt(a**2 + b**2)
    threshold = Z[np.argmax(distance), 2]

    return scipy.cluster.hierarchy.fcluster(Z, threshold, criterion="distance")
