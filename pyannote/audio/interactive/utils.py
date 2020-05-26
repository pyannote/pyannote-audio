# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHORS
# Hervé Bredin - http://herve.niderb.fr


from typing import List, Tuple

Time = float
from pyannote.core import SlidingWindow
import numpy as np


def time2index(
    constraints_time: List[Tuple[Time, Time]], window: SlidingWindow,
) -> List[Tuple[int, int]]:
    """Convert time-based constraints to index-based constraints

    Parameters
    ----------
    constraints_time : list of (float, float)
        Time-based constraints
    window : SlidingWindow
        Window used for embedding extraction

    Returns
    -------
    constraints : list of (int, int)
        Index-based constraints
    """

    constraints = []
    for t1, t2 in constraints_time:
        i1 = window.closest_frame(t1)
        i2 = window.closest_frame(t2)
        if i1 == i2:
            continue
        constraints.append((i1, i2))
    return constraints


def index2index(
    constraints: List[Tuple[int, int]],
    keep: np.ndarray,
    reverse=False,
    return_mapping=False,
) -> List[Tuple[int, int]]:
    """Map constraints from original to keep-only index base

    Parameters
    ----------
    constraints : list of pairs
        Constraints in original index base.
    keep : np.ndarray
        Boolean array indicating whether to keep observations.
    reverse : bool
        Set to True to go from keep-only to original index base.
    return_mapping : bool, optional
        Return mapping instead of mapped constraints.

    Returns
    -------
    shifted_constraints : list of index pairs
        Constraints in keep-only index base.
    """

    if reverse:
        mapping = np.arange(len(keep))[keep]
    else:
        mapping = np.cumsum(keep) - 1

    if return_mapping:
        return mapping

    return [
        (mapping[i1], mapping[i2]) for i1, i2 in constraints if keep[i1] and keep[i2]
    ]
