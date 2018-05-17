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


import warnings
import numpy as np

import python_speech_features
from pyannote.audio.features.utils import read_audio


from pyannote.core.segment import SlidingWindow
from pyannote.core.feature import SlidingWindowFeature
from pyannote.audio.features.utils import PyannoteFeatureExtractionError
from pyannote.database.util import get_unique_identifier


class PySpeechFeaturesExtractor(object):
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

    def __init__(self, sample_rate=16000, duration=0.025, step=0.01):

        super(PySpeechFeaturesExtractor, self).__init__()

        self.sample_rate = sample_rate
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

        # --- load audio file
        y, sample_rate = read_audio(item,
                                    sample_rate=self.sample_rate,
                                    mono=True)

        data = self.process(y, sample_rate)

        if np.any(np.isnan(data)):
            uri = get_unique_identifier(item)
            msg = 'Features extracted from "{uri}" contain NaNs.'
            warnings.warn(msg.format(uri=uri))

        return SlidingWindowFeature(data, self.sliding_window_)


class PySpeechFeaturesMFCC(PySpeechFeaturesExtractor):
    """MFCC with python_speech_features

    Parameters
    ----------
    sample_rate : int, optional
        Sampling rate.
    duration : float, optional
        Defaults to 0.025 (25ms)
    step : float, optional
        Defaults to 0.01 (10ms)
    coefs : int, optional
        Number of coefficients. Defaults to 13.
    """

    def __init__(self, sample_rate=16000, duration=0.025, step=0.01,
                 coefs=13):

        super(PySpeechFeaturesMFCC, self).__init__(
            sample_rate=sample_rate, duration=duration, step=step)

        self.coefs = coefs

    def process(self, y, sample_rate):

        return python_speech_features.mfcc(
            y, samplerate=sample_rate, winlen=self.duration, winstep=self.step,
            numcep=self.coefs)

    def dimension(self):
        return self.coefs
