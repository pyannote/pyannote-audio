# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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


import warnings

import matplotlib
import numpy as np
from numpy.typing import ArrayLike

from pyannote.metrics.binary_classification import det_curve, precision_recall_curve

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_distributions(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    xlim: tuple[float, float] | None = None,
    nbins: int = 100,
    ymax: float = 3.0,
    dpi: int = 150,
) -> bool:
    """Scores distributions

    This function will create (and overwrite) the following files:
        - {save_to}.scores.png
        - {save_to}.scores.eps

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    save_to : str
        Files path prefix
    """

    plt.figure(figsize=(12, 12))

    if xlim is None:
        xlim = (np.min(scores), np.max(scores))

    bins = np.linspace(xlim[0], xlim[1], nbins)
    plt.hist(scores[y_true], bins=bins, color="g", alpha=0.5, density=True)
    plt.hist(scores[~y_true], bins=bins, color="r", alpha=0.5, density=True)

    # TODO heuristic to estimate ymax from nbins and xlim
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(save_to + ".scores.png", dpi=dpi)
    plt.savefig(save_to + ".scores.eps")
    plt.close()

    return True


def plot_det_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    distances: bool = False,
    dpi: int = 150,
) -> float:
    """DET curve

    This function will create (and overwrite) the following files:
        - {save_to}.det.png
        - {save_to}.det.eps
        - {save_to}.det.txt

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    save_to : str
        Files path prefix.
    distances : boolean, optional
        When True, indicate that `scores` are actually `distances`
    dpi : int, optional
        Resolution of .png file. Defaults to 150.

    Returns
    -------
    eer : float
        Equal error rate
    """

    fpr, fnr, thresholds, eer = det_curve(y_true, scores, distances=distances)

    # plot DET curve
    plt.figure(figsize=(12, 12))
    plt.loglog(fpr, fnr, "b")
    plt.loglog([eer], [eer], "bo")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.xlim(1e-2, 1.0)
    plt.ylim(1e-2, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_to + ".det.png", dpi=dpi)
    plt.savefig(save_to + ".det.eps")
    plt.close()

    # save DET curve in text file
    txt = save_to + ".det.txt"
    line = "{t:.6f} {fp:.6f} {fn:.6f}\n"
    with open(txt, "w") as f:
        f.writelines(
            line.format(t=t, fp=fp, fn=fn)
            for i, (t, fp, fn) in enumerate(zip(thresholds, fpr, fnr))
        )

    return eer


def plot_precision_recall_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    save_to: str,
    distances: bool = False,
    dpi: int = 150,
) -> float:
    """Precision/recall curve

    This function will create (and overwrite) the following files:
        - {save_to}.precision_recall.png
        - {save_to}.precision_recall.eps
        - {save_to}.precision_recall.txt

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    save_to : str
        Files path prefix.
    distances : boolean, optional
        When True, indicate that `scores` are actually `distances`
    dpi : int, optional
        Resolution of .png file. Defaults to 150.

    Returns
    -------
    auc : float
        Area under precision/recall curve
    """

    precision, recall, thresholds, auc = precision_recall_curve(
        y_true, scores, distances=distances
    )

    # plot P/R curve
    plt.figure(figsize=(12, 12))
    plt.plot(recall, precision, "b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_to + ".precision_recall.png", dpi=dpi)
    plt.savefig(save_to + ".precision_recall.eps")
    plt.close()

    # save P/R curve in text file
    txt = save_to + ".precision_recall.txt"
    line = "{t:.6f} {p:.6f} {r:.6f}\n"
    with open(txt, "w") as f:
        f.writelines(
            line.format(t=t, p=p, r=r)
            for i, (t, p, r) in enumerate(zip(thresholds, precision, recall))
        )

    return auc
