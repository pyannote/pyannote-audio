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
# Mathieu Bernard
# Julien Karadayi
# Marvin Lavechin

"""
Feature extraction with Shennong
--------------------------------
"""

from shennong.audio import Audio
from shennong.features.processor.mfcc import MfccProcessor
import numpy as np

from .base import FeatureExtraction
from pyannote.core.segment import SlidingWindow
from shennong.features.pipeline import get_default_config, extract_features
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.processor.pitch import (
    PitchProcessor, PitchPostProcessor)

class ShennongFeatureExtraction(FeatureExtraction):
    """Shennong feature extraction base class

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
    with_cmvn : bool, optional
        Defaults to False
    with_pitch: bool, optional
        Defaults to False
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01):

        super().__init__(sample_rate=sample_rate,
                         augmentation=augmentation)
        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(start=-.5*self.duration,
                                             duration=self.duration,
                                             step=self.step)

    def get_sliding_window(self):
        return self.sliding_window_

class ShennongMfccPitch(ShennongFeatureExtraction):
    """Shennong MFCC

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
            |pitch1|
            |pitch2|  Coefficients of pitch estimation
            |pitch3|
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
        Number of coefficients. Defaults to 13.
    fmin : int, optional
        min frequency for pitch estimation. Defaults to 20.
    fmax : int, optional
        max frequency for pitch estimation. Defaults to 500.
    D : bool, optional
        Add first order derivatives. Defaults to True.
    DD : bool, optional
        Add second order derivatives. Defaults to True.
    with_pitch: bool, optional
        Compute Pitch Estimation (w/ same step and Duration as MFCC).
        Defaults to True.

    Notes
    -----
    Internal setup
        * fftWindow = Hanning
        * melMaxFreq = sampleFreq / 2 - 100
        * melMinFreq = 20
        * melNbFilters = 40


    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01,
                 e=False, De=True, DDe=True,
                 coefs=13, D=True, DD=True,
                 fmin=20, fmax=500, n_mels=40,
                 with_pitch=True):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD
        self.with_pitch = with_pitch

        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def get_context_duration(self):
        return 0.

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
        # force y to be of shape (n,) if shape is (n,1)
        #if y.shape[1] == 1:
        #    y = y.reshape(y.shape[0])
        #y = y.astype('float64')
        # create audio object for shennong
        audio = Audio(data=y, sample_rate=sample_rate)
        #audio = Audio.load(fin)
        #sample_rate = audio.sample_rate
        # MFCC parameters
        processor = MfccProcessor(sample_rate=sample_rate)
        processor.window_type = 'hanning'
        processor.low_freq = 20
        processor.high_freq = -100 # defines it as (nyquist - 100)
        processor.use_energy = True

        # MFCC extraction
        #audio = Audio(data=y, sample_rate=sample_rate)
        mfcc = processor.process(audio)
        print("coucou")
        # compute deltas
        if self.D:
            # define first or second order derivative
            if not self.DD:
                derivative_proc = DeltaPostProcessor(order=1)
            else:
                derivative_proc = DeltaPostProcessor(order=2)

            # process Mfccs
            mfcc = derivative_proc.process(mfcc)

        # Compute Pitch
        if self.with_pitch:
            print("pitchohmonpitch")
            # define pitch estimation parameters
            processor = PitchProcessor(frame_shift=self.step,
                                       frame_length=self.duration)
            processor.sample_rate = sample_rate
            processor.min_f0 = self.fmin
            processor.max_f0 = self.fmax

            # estimate pitch
            pitch = processor.process(audio)
            print('have pitch, concatenating...')

            # concatenate mfcc w/pitch
            mfcc = mfcc.concatenate(pitch, 5)

        print("just before returning")
        return mfcc.data

    def get_dimension(self):
        n_features = 0
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD
        n_features += self.with_pitch * 2
        return n_features
