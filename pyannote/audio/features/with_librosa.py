#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

import matplotlib
matplotlib.use('Agg')
import librosa

import pysndfile.sndio
from pyannote.core.segment import SlidingWindow
from pyannote.core.feature import SlidingWindowFeature
from pyannote.audio.features.utils import PyannoteFeatureExtractionError
from pyannote.database.util import get_unique_identifier


class LibrosaFeatureExtractor(object):
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

    def __init__(self, duration=0.025, step=0.01):

        super(LibrosaFeatureExtractor, self).__init__()

        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(start=-.5*self.duration,
                                             duration=self.duration,
                                             step=self.step)

    def sliding_window(self):
        return self.sliding_window_

    def dimension(self):
        raise NotImplementedError('')

    def __call__(self, item):
        """Extract features

        Parameters
        ----------
        item : dict

        Returns
        -------
        features : SlidingWindowFeature

        """

        try:
            wav = item['wav']
            y, sample_rate, encoding = pysndfile.sndio.read(wav)
        except IOError as e:
            raise PyannoteFeatureExtractionError(e.message)

        if np.any(np.isnan(y)):
            uri = get_unique_identifier(item)
            msg = 'pysndfile output contains NaNs for file "{uri}".'
            raise PyannoteFeatureExtractionError(msg.format(uri=uri))

        # reshape before selecting channel
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        channel = item.get('channel', 1)
        y = y[:, channel - 1]

        data = self.process(y, sample_rate)

        if np.any(np.isnan(data)):
            uri = get_unique_identifier(item)
            msg = 'Features extracted from "{uri}" contain NaNs.'
            warnings.warn(msg.format(uri=uri))

        return SlidingWindowFeature(data.T, self.sliding_window_)


class LibrosaRMSE(LibrosaFeatureExtractor):
    """

    Parameters
    ----------
    duration : float, optional
        Defaults to 0.025 (25ms)
    step : float, optional
        Defaults to 0.01 (10ms)

    """

    def process(self, y, sample_rate):
        n_fft = int(self.duration * sample_rate)
        hop_length = int(self.step * sample_rate)
        return librosa.feature.rmse(y=y, n_fft=n_fft, hop_length=hop_length)

    def dimension(self):
        return 1


class LibrosaMFCC(LibrosaFeatureExtractor):
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
        Defaults to 0.025 (25ms)
    step : float, optional
        Defaults to 0.01 (10ms)
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
    Internal setup
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(self, duration=0.025, step=0.01,
                 e=False, De=True, DDe=True,
                 coefs=19, D=True, DD=True,
                 fmin=0.0, fmax=None, n_mels=40):

        super(LibrosaMFCC, self).__init__(duration=duration, step=step)

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

        self.n_mels = n_mels  # yaafe / 40
        self.fmin = fmin      # yaafe / 130.0
        self.fmax = fmax      # yaafe / 6854.0

    def process(self, y, sample_rate):

        # adding because C0 is the energy
        n_mfcc = self.coefs + 1

        n_fft = int(self.duration * sample_rate)
        hop_length = int(self.step * sample_rate)

        mfcc = librosa.feature.mfcc(
            y=y, sr=sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=self.n_mels, htk=True,
            fmin=self.fmin, fmax=self.fmax)

        if self.De or self.D:
            mfcc_d = librosa.feature.delta(
                mfcc, width=9, order=1, axis=-1, trim=True)

        if self.DDe or self.DD:
            mfcc_dd = librosa.feature.delta(
                mfcc, width=9, order=2, axis=-1, trim=True)

        stack = []

        if self.e:
            stack.append(mfcc[0, :])

        stack.append(mfcc[1:, :])

        if self.De:
            stack.append(mfcc_d[0, :])

        if self.D:
            stack.append(mfcc_d[1:, :])

        if self.DDe:
            stack.append(mfcc_dd[0, :])

        if self.DD:
            stack.append(mfcc_dd[1:, :])

        return np.vstack(stack)

    def dimension(self):

        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD

        return n_features
