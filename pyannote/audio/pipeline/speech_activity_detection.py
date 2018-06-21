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

import numpy as np
from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature
from pyannote.audio.features import Precomputed
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.audio.signal import Binarize
from .base import Pipeline
import chocolate


class SpeechActivityDetection(Pipeline):
    """Speech activity detection pipeline

    Parameters
    ----------
    precomputed : str
        Path to precomputed SAD scores.
    """

    def __init__(self, precomputed=None, **kwargs):
        super(SpeechActivityDetection, self).__init__()
        self.precomputed = precomputed

        self.precomputed_ = Precomputed(self.precomputed)
        self.has_overlap_ = self.precomputed_.dimension() == 3

        self.with_params(**kwargs)

    def get_tune_space(self):

        space = {
            'speech_onset': chocolate.uniform(0., 1.),
            'speech_offset': chocolate.uniform(0., 1.),
            'speech_min_duration_on': chocolate.uniform(0., 2.),
            'speech_min_duration_off': chocolate.uniform(0., 2.),
            'speech_pad_onset': chocolate.uniform(-1., 1.),
            'speech_pad_offset': chocolate.uniform(-1., 1.)
        }

        if self.has_overlap_:
            space.update({
                'overlap_onset': chocolate.uniform(0., 1.),
                'overlap_offset': chocolate.uniform(0., 1.),
                'overlap_min_duration_on': chocolate.uniform(0., 2.),
                'overlap_min_duration_off': chocolate.uniform(0., 2.),
                'overlap_pad_onset': chocolate.uniform(-1., 1.),
                'overlap_pad_offset': chocolate.uniform(-1., 1.)
            })

        return space

    def get_tune_metric(self):
        return DetectionErrorRate()

    def with_params(self, **params):

        # initialize speech/non-speech binarizer
        speech_params = {
            '_'.join(param.split('_')[1:]): value
            for param, value in params.items()
            if param.startswith('speech_')}
        self.speech_binarize_ = Binarize(**speech_params)

        # initialize overlap binarizer
        if self.has_overlap_:
            overlap_params = {
                '_'.join(param.split('_')[1:]): value
                for param, value in params.items()
                if param.startswith('overlap_')}
            self.overlap_binarize_ = Binarize(**overlap_params)

        return self

    def apply(self, current_file):

        # extract precomputed scores
        precomputed = self.precomputed_(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(precomputed.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(precomputed.data) if self.log_scale_ \
               else precomputed.data

        # speech vs. non-speech
        speech_prob = SlidingWindowFeature(
            1. - data[:, 0],
            precomputed.sliding_window)
        speech = self.speech_binarize_.apply(speech_prob)

        if self.has_overlap_:

            # overlap vs. non-overlap
            overlap_prob = SlidingWindowFeature(
                data[:, 2], precomputed.sliding_window)
            overlap = self.overlap_binarize_.apply(overlap_prob)

            # overlap speech can only happen in speech regions
            overlap = overlap.crop(speech)
        else:
            # empty timeline
            overlap = Timeline()

        speech = speech.to_annotation(generator='string')
        overlap = overlap.to_annotation(generator='int')
        hypothesis = speech.update(overlap)

        return hypothesis
