#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2019 CNRS

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
Feature extraction using [`librosa`](https://librosa.github.io/librosa/)
"""

import librosa
import numpy as np

from .base import FeatureExtraction
from pyannote.core.segment import SlidingWindow


class LibrosaFeatureExtraction(FeatureExtraction):
    """librosa feature extraction base class

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025 (25ms).
    step : float, optional
        Defaults to 0.010 (10ms).
    """

    def __init__(self, sample_rate=16000, augmentation=None, duration=0.025, step=0.01):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation)
        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(
            start=-0.5 * self.duration, duration=self.duration, step=self.step
        )

    def get_resolution(self):
        return self.sliding_window_


class LibrosaSpectrogram(LibrosaFeatureExtraction):
    """librosa spectrogram

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    """

    def __init__(
        self, sample_rate=16000, augmentation=None, duration=0.025, step=0.010
    ):

        super().__init__(
            sample_rate=sample_rate,
            augmentation=augmentation,
            duration=duration,
            step=step,
        )

        self.n_fft_ = int(self.duration * self.sample_rate)
        self.hop_length_ = int(self.step * self.sample_rate)

    def get_dimension(self):
        return self.n_fft_ // 2 + 1

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """

        fft = librosa.core.stft(
            y=y.squeeze(),
            n_fft=self.n_fft_,
            hop_length=self.hop_length_,
            center=True,
            window="hamming",
        )
        return np.abs(fft).T


class LibrosaMelSpectrogram(LibrosaFeatureExtraction):
    """librosa mel-spectrogram

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    n_mels : int, optional
        Defaults to 96.
    """

    def __init__(
        self,
        sample_rate=16000,
        augmentation=None,
        duration=0.025,
        step=0.010,
        n_mels=96,
    ):

        super().__init__(
            sample_rate=sample_rate,
            augmentation=augmentation,
            duration=duration,
            step=step,
        )

        self.n_mels = n_mels
        self.n_fft_ = int(self.duration * self.sample_rate)
        self.hop_length_ = int(self.step * self.sample_rate)

    def get_dimension(self):
        return self.n_mels

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_mels) numpy array
            Features
        """

        X = librosa.feature.melspectrogram(
            y.squeeze(),
            sr=sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft_,
            hop_length=self.hop_length_,
            power=2.0,
        )

        return librosa.amplitude_to_db(X, ref=1.0, amin=1e-5, top_db=80.0).T


class LibrosaMFCC(LibrosaFeatureExtraction):
    """librosa MFCC

    ::

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
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Data augmentation.
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
    Internal setup
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(
        self,
        sample_rate=16000,
        augmentation=None,
        duration=0.025,
        step=0.01,
        e=False,
        De=True,
        DDe=True,
        coefs=19,
        D=True,
        DD=True,
        fmin=0.0,
        fmax=None,
        n_mels=40,
    ):

        super().__init__(
            sample_rate=sample_rate,
            augmentation=augmentation,
            duration=duration,
            step=step,
        )

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

        self.n_mels = n_mels  # yaafe / 40
        self.fmin = fmin  # yaafe / 130.0
        self.fmax = fmax  # yaafe / 6854.0

    def get_context_duration(self):
        return 0.0

    def get_features(self, y, sample_rate):
        """Feature extraction

        Parameters
        ----------
        y : (n_samples, 1) numpy array
            Waveform
        sample_rate : int
            Sample rate

        Returns
        -------
        data : (n_frames, n_dimensions) numpy array
            Features
        """

        # adding because C0 is the energy
        n_mfcc = self.coefs + 1

        n_fft = int(self.duration * sample_rate)
        hop_length = int(self.step * sample_rate)

        mfcc = librosa.feature.mfcc(
            y=y.squeeze(),
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels,
            htk=True,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        if self.De or self.D:
            mfcc_d = librosa.feature.delta(mfcc, width=9, order=1, axis=-1)

        if self.DDe or self.DD:
            mfcc_dd = librosa.feature.delta(mfcc, width=9, order=2, axis=-1)

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

        return np.vstack(stack).T

    def get_dimension(self):
        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD
        return n_features
