#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2017 CNRS

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

from __future__ import unicode_literals

import warnings
import numpy as np
import yaafelib
import pysndfile.sndio

from pyannote.core.segment import SlidingWindow
from pyannote.core.feature import SlidingWindowFeature
from pyannote.audio.features.utils import PyannoteFeatureExtractionError
from pyannote.database.util import get_unique_identifier


class YaafeFeatureExtractor(object):
    """

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    block_size : int, optional
        Defaults to 512.
    step_size : int, optional
        Defaults to 256.

    """

    def __init__(self, duration=0.025, step=0.010, num_concatenate=1):

        super(YaafeFeatureExtractor, self).__init__()

        self.duration = duration
        self.step = step
        self.num_concatenate = num_concatenate

        start = -0.5 * self.duration
        self.sliding_window_ = SlidingWindow(start=start,
                                             duration=self.duration,
                                             step=self.step)

        self.engine_ = yaafelib.Engine()

    def dimension(self):
        raise NotImplementedError('')

    def sliding_window(self):
        return self.sliding_window_

    def __call__(self, item):
        """Extract features

        Parameters
        ----------
        item : dict

        Returns
        -------
        features : SlidingWindowFeature

        """

        # --- load audio file
        try:
            wav = item['wav']
            y, sample_rate, encoding = pysndfile.sndio.read(wav)
        except IOError as e:
            raise PyannoteFeatureExtractionError(e)

        # --- update data_flow every time sample rate changes
        if not hasattr(self, 'sample_rate_') or self.sample_rate_ != sample_rate:
            self.sample_rate_ = sample_rate
            feature_plan = yaafelib.FeaturePlan(sample_rate=self.sample_rate_)
            for name, recipe in self.definition():
                assert feature_plan.addFeature(
                    "{name}: {recipe}".format(name=name, recipe=recipe))
            data_flow = feature_plan.getDataFlow()
            self.engine_.load(data_flow)

        # reshape before selecting channel
        if len(y.shape) < 2:
            y = y.reshape((-1, 1))

        channel = item.get('channel', 1)
        y = y[:, channel - 1]

        # Yaafe needs this: float64, column-contiguous, 2-dimensional
        y = np.array(y, dtype=np.float64, order='C').reshape((1, -1))

        # --- extract features
        features = self.engine_.processAudio(y)
        data = np.hstack([features[name] for name, _ in self.definition()])
        a,b = data.shape
        num_padding = self.num_concatenate//2
        if self.num_concatenate % 2 == 0:
            expanded_data = np.concatenate((np.zeros((num_padding,b))+data[0],data,np.zeros((num_padding-1,b))+data[-1]))
        else:
            expanded_data = np.concatenate((np.zeros((num_padding,b))+data[0],data,np.zeros((num_padding,b))+data[-1]))
        data = np.lib.stride_tricks.as_strided(expanded_data, shape=(a, b*self.num_concatenate), strides=data.strides)        

        self.engine_.reset()

        # --- return as SlidingWindowFeature
        if np.any(np.isnan(data)):
            uri = get_unique_identifier(item)
            msg = 'Features extracted from "{uri}" contain NaNs.'
            warnings.warn(msg.format(uri=uri))

        return SlidingWindowFeature(data, self.sliding_window_)


class YaafeCompound(YaafeFeatureExtractor):

    def __init__(self, extractors, duration=0.025, step=0.010):

        assert all(e.duration == duration for e in extractors)
        assert all(e.step == step for e in extractors)

        super(YaafeCompound, self).__init__(duration=duration, step=step)

        self.extractors = extractors

    def dimension(self):
        return sum(extractor.dimension() for extractor in self.extractors)

    def definition(self):
        return [(name, recipe)
                for e in self.extractors for name, recipe in e.definition()]

    def __hash__(self):
        return hash(tuple(self.definition()))


class YaafeZCR(YaafeFeatureExtractor):

    def dimension(self):
        return 1

    def definition(self):

        blockSize = int(self.sample_rate_ * self.duration)
        stepSize = int(self.sample_rate_ * self.step)

        d = [(
            "zcr",
            "ZCR blockSize=%d stepSize=%d" % (blockSize, stepSize)
        )]

        return d


class YaafeMFCC(YaafeFeatureExtractor):
    """
        | e    |  energy
        | c1   |
        | c2   |  coefficients
        | c3   |
        | ...  |
        | Δe   |  energy first derivative
        | Δc1  |
    x = | Δc2  |  coefficients first derivatives
        | Δc3  |
        | ...  |
        | ΔΔe  |  energy second derivative
        | ΔΔc1 |
        | ΔΔc2 |  coefficients second derivatives
        | ΔΔc3 |
        | ...  |


    Parameters
    ----------

    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.

    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 11.
    De : bool, optional
        Keep energy first derivative. Defaults to False.
    D : bool, optional
        Add first order derivatives. Defaults to False.
    DDe : bool, optional
        Keep energy second derivative. Defaults to False.
    DD : bool, optional
        Add second order derivatives. Defaults to False.

    Notes
    -----
    Default Yaafe values:
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(
        self, duration=0.025, step=0.010, num_concatenate=1,
        e=True, coefs=11, De=False, DDe=False, D=False, DD=False,
    ):

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

        super(YaafeMFCC, self).__init__(duration=duration, step=step, num_concatenate=num_concatenate)

    def dimension(self):

        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD

        return n_features*self.num_concatenate

    def definition(self):

        blockSize = int(self.sample_rate_ * self.duration)
        stepSize = int(self.sample_rate_ * self.step)

        d = []

        # --- coefficients
        # 0 if energy is kept
        # 1 if energy is removed
        d.append((
            "mfcc",
            "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
            "blockSize=%d stepSize=%d" % (
                0 if self.e else 1,
                self.coefs + self.e * 1,
                blockSize, stepSize
            )))

        # --- 1st order derivatives
        if self.De or self.D:
            d.append((
                "mfcc_d",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=1" % (
                    0 if self.De else 1,
                    self.D * self.coefs + self.De * 1,
                    blockSize, stepSize
                )))

        # --- 2nd order derivatives
        if self.DDe or self.DD:
            d.append((
                "mfcc_dd",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=2" % (
                    0 if self.DDe else 1,
                    self.DD * self.coefs + self.DDe * 1,
                    blockSize, stepSize
                )))

        return d
