#!/usr/bin/env python
# encoding: utf-8

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
Feature extraction with python_speech_features
----------------------------------------------
"""

import python_speech_features
import numpy as np

from .base import FeatureExtraction
from pyannote.core.segment import SlidingWindow


class PySpeechFeaturesExtraction(FeatureExtraction):
    """python_speech_features base class

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Defaults to no augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation)
        self.duration = duration
        self.step = step

        self.sliding_window_ = SlidingWindow(start=-.5*self.duration,
                                             duration=self.duration,
                                             step=self.step)

    def get_sliding_window(self):
        return self.sliding_window_


class PySpeechFeaturesMFCC(PySpeechFeaturesExtraction):
    """MFCC with python_speech_features

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    augmentation : `pyannote.audio.augmentation.Augmentation`, optional
        Defaults to no augmentation.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    coefs : int, optional
        Number of coefficients. Defaults to 13.
    """

    def __init__(self, sample_rate=16000, augmentation=None,
                 duration=0.025, step=0.01, coefs=13):

        super().__init__(sample_rate=sample_rate, augmentation=augmentation,
                         duration=duration, step=step)
        self.coefs = coefs

    def get_dimension(self):
        return self.coefs

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

        return python_speech_features.mfcc(
            32768 * y, samplerate=sample_rate,
            winlen=self.duration, winstep=self.step,
            numcep=self.coefs)
