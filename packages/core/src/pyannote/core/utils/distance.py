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

import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance


def l2_normalize(X: np.ndarray):
    """L2 normalize vectors

    Parameters
    ----------
    X : `np.ndarray`
        (n_samples, n_dimensions) vectors.

    Returns
    -------
    normalized : `np.ndarray`
        (n_samples, n_dimensions) L2-normalized vectors

    """

    norm = np.sqrt(np.sum(X**2, axis=1))
    norm[norm == 0] = 1.0
    return (X.T / norm).T


def dist_range(metric="euclidean", normalize=False):
    """Return range of possible distance between two vectors

    Parameters
    ----------
    metric : `str`, optional
        Metric. Defaults to 'euclidean'
    normalize : `bool`, optional
        Set to True if vectors are L2-normalized. Defaults to False.

    Returns
    -------
    min_dist, max_dist : `float`
        Range of possible distance.
    """

    if metric == "euclidean":
        if normalize:
            return (0.0, 2.0)
        return (0.0, np.inf)

    if metric == "sqeuclidean":
        if normalize:
            return (0.0, 4.0)
        return (0.0, np.inf)

    if metric == "cosine":
        return (0.0, 2.0)

    if metric == "angular":
        return (0.0, np.pi)

    msg = f"dist_range does not support {metric} metric."
    raise NotImplementedError(msg)


def _pdist_func_1D(X, func):
    """Helper function for pdist"""

    (n_items,) = X.shape

    if n_items < 2:
        return np.array([])

    distances = []

    for i in range(n_items - 1):
        distance = func(X[i], X[i + 1 :])
        distances.append(distance)

    return np.hstack(distances)


def pdist(fX, metric="euclidean", **kwargs):
    """Same as scipy.spatial.distance with support for additional metrics

    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == "angular":
        cosine = scipy.spatial.distance.pdist(fX, metric="cosine", **kwargs)
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == "equal":
        assert fX.ndim == 1, f"'{metric}' metric only supports 1-dimensional fX."
        return _pdist_func_1D(fX, lambda x, X: x == X)

    elif metric == "minimum":
        assert fX.ndim == 1, f"'{metric}' metric only supports 1-dimensional fX."
        return _pdist_func_1D(fX, np.minimum)

    elif metric == "maximum":
        assert fX.ndim == 1, f"'{metric}' metric only supports 1-dimensional fX."
        return _pdist_func_1D(fX, np.maximum)

    elif metric == "average":
        assert fX.ndim == 1, f"'{metric}' metric only supports 1-dimensional fX."
        return _pdist_func_1D(fX, lambda x, X: 0.5 * (x + X))

    else:
        return scipy.spatial.distance.pdist(fX, metric=metric, **kwargs)


def _cdist_func_1D(X_trn, X_tst, func):
    """Helper function for cdist"""
    return np.vstack(func(x_trn, X_tst) for x_trn in iter(X_trn))


def cdist(fX_trn, fX_tst, metric="euclidean", **kwargs):
    """Same as scipy.spatial.distance.cdist with support for additional metrics

    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == "angular":
        cosine = scipy.spatial.distance.cdist(fX_trn, fX_tst, metric="cosine", **kwargs)
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == "equal":
        assert fX_trn.ndim == 1 and fX_tst.ndim == 1, (
            f"'{metric}' metric only supports 1-dimensional fX_trn and fX_tst."
        )
        return _cdist_func_1D(fX_trn, fX_tst, lambda x_trn, X_tst: x_trn == X_tst)

    elif metric == "minimum":
        assert fX_trn.ndim == 1 and fX_tst.ndim == 1, (
            f"'{metric}' metric only supports 1-dimensional fX_trn and fX_tst."
        )
        return _cdist_func_1D(fX_trn, fX_tst, np.minimum)

    elif metric == "maximum":
        assert fX_trn.ndim == 1 and fX_tst.ndim == 1, (
            f"'{metric}' metric only supports 1-dimensional fX_trn and fX_tst."
        )
        return _cdist_func_1D(fX_trn, fX_tst, np.maximum)

    elif metric == "average":
        assert fX_trn.ndim == 1 and fX_tst.ndim == 1, (
            f"'{metric}' metric only supports 1-dimensional fX_trn and fX_tst."
        )
        return _cdist_func_1D(
            fX_trn, fX_tst, lambda x_trn, X_tst: 0.5 * (x_trn + X_tst)
        )

    else:
        return scipy.spatial.distance.cdist(fX_trn, fX_tst, metric=metric, **kwargs)


def to_condensed(n, i, j):
    """Compute index in condensed pdist matrix

                V
        0 | . 0 1 2 3
     -> 1 | . . 4 5 6 <-   ==>   0 1 2 3 4 5 6 7 8 9
        2 | . . . 7 8                    ^
        3 | . . . . 9
        4 | . . . . .
           ----------
            0 1 2 3 4

    Parameters
    ----------
    n : int
        Number of inputs in squared pdist matrix
    i, j : `int` or `numpy.ndarray`
        Indices in squared pdist matrix

    Returns
    -------
    k : `int` or `numpy.ndarray`
        Index in condensed pdist matrix
    """
    i, j = np.array(i), np.array(j)
    if np.any(i == j):
        raise ValueError("i and j should be different.")
    i, j = np.minimum(i, j), np.maximum(i, j)
    return np.int64(i * n - i * i / 2 - 3 * i / 2 + j - 1)


def to_squared(n, k):
    """Compute indices in squared matrix

    Parameters
    ----------
    n : int
        Number of inputs in squared pdist matrix
    k : `int` or `numpy.ndarray`
        Index in condensed pdist matrix

    Returns
    -------
    i, j : `int` or `numpy.ndarray`
        Indices in squared pdist matrix

    """
    k = np.array(k)
    i = np.int64(n - np.sqrt(-8 * k + 4 * n**2 - 4 * n + 1) / 2 - 1 / 2)
    j = np.int64(i**2 / 2 - i * n + 3 * i / 2 + k + 1)
    return i, j
