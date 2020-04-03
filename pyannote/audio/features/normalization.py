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


import numpy as np
import pandas as pd
from pyannote.core import SlidingWindowFeature


class GlobalStandardization(object):
    """Mean/variance normalization"""

    def get_context_duration(self):
        return 0.0

    def __call__(self, features, sliding_window=None):
        """Apply global standardization

        Parameters
        ----------
        features : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        sliding_window : `SlidingWindow`, optional
            Not used.

        Returns
        -------
        normalized : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Standardized features
        """

        if isinstance(features, SlidingWindowFeature):
            data = features.data
        else:
            data = features

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0, ddof=1)
        sigma[sigma == 0.0] = 1e-6

        normalized = (data - mu) / sigma

        if isinstance(features, SlidingWindowFeature):
            return SlidingWindowFeature(normalized, features.sliding_window)
        else:
            return normalized


class ShortTermStandardization(object):
    """Short term mean/variance normalization

    Parameters
    ----------
    duration : float
        Window duration in seconds.
    """

    def __init__(self, duration=3.0):
        super(ShortTermStandardization, self).__init__()
        self.duration = duration

    def get_context_duration(self):
        return 0.5 * self.duration

    def __call__(self, features, sliding_window=None):
        """Apply short-term standardization

        Parameters
        ----------
        features : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Features.
        sliding_window : `SlidingWindow`, optional
            Sliding window when `features` is a `numpy.ndarray`.
            Not used when `features` is a `SlidingWindowFeature` instance.

        Returns
        -------
        normalized : `SlidingWindowFeature` or (n_samples, n_features ) `numpy.ndarray`
            Standardized features
        """

        if isinstance(features, SlidingWindowFeature):
            features_ = features
        else:
            features_ = SlidingWindowFeature(features, sliding_window)

        window = features_.sliding_window.samples(self.duration, mode="center")
        if not window % 2:
            window += 1

        rolling = pd.DataFrame(features_.data).rolling(
            window=window, center=True, min_periods=window
        )
        mu = np.array(rolling.mean())
        sigma = np.array(rolling.std(ddof=1))

        for i in range(window // 2):

            data = features_.data[: i + window // 2 + 1, :]
            mu[i] = np.mean(data, axis=0)
            sigma[i] = np.std(data, axis=0, ddof=1)

            data = features_.data[-i - window // 2 - 1 :, :]
            mu[-i - 1] = np.mean(data, axis=0)
            sigma[-i - 1] = np.std(data, axis=0, ddof=1)

        sigma[sigma == 0.0] = 1e-6

        normalized_ = (features_.data - mu) / sigma

        if isinstance(features, SlidingWindowFeature):
            return SlidingWindowFeature(normalized_, features.sliding_window)
        else:
            return normalized_
