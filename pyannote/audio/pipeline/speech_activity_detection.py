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

from typing import Optional
from pathlib import Path
import numpy as np

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.audio.signal import Binarize
from pyannote.audio.features import Precomputed

from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionErrorRate

from pyannote.database import get_annotated


class OracleSpeechActivityDetection(Pipeline):
    """Oracle speech activity detection"""

    def __call__(self, current_file: dict) -> Annotation:
        """Return groundtruth speech activity detection

        Parameter
        ---------
        current_file : `dict`
            Dictionary as provided by `pyannote.database`.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speech regions
        """

        speech = current_file['annotation'].get_timeline().support()
        return speech.to_annotation(generator='string', modality='speech')


class SpeechActivityDetection(Pipeline):
    """Speech activity detection pipeline

    Parameters
    ----------
    scores : `Path`, optional
        Path to precomputed scores on disk.
    """

    def __init__(self, scores: Optional[Path] = None,
                 scores_name: Optional[str] = 'sad_scores',
                 detection: Optional[bool] = True):
        super().__init__()

        self.scores = scores
        self.scores_name = scores_name
        self.detection = detection

        if self.scores is not None:
            self._precomputed = Precomputed(self.scores)

        # hyper-parameters
        self.onset = Uniform(0., 1.)
        self.offset = Uniform(0., 1.)
        self.min_duration_on = Uniform(0., 2.)
        self.min_duration_off = Uniform(0., 2.)
        self.pad_onset = Uniform(-1., 1.)
        self.pad_offset = Uniform(-1., 1.)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            pad_onset=self.pad_onset,
            pad_offset=self.pad_offset)

    def __call__(self, current_file: dict) -> Annotation:
        """Apply speech activity detection

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol. May contain a
            'sad_scores' key providing precomputed scores.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Speech regions.
        """

        # precomputed SAD scores
        sad_scores = current_file.get(self.scores_name)
        if sad_scores is None:
            sad_scores = self._precomputed(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(sad_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(sad_scores.data) if self.log_scale_ \
               else sad_scores.data

        # speech vs. non-speech
        if data.shape[1] > 1:
            speech_prob = SlidingWindowFeature(1. - data[:, 0], sad_scores.sliding_window)
        else:
            speech_prob = SlidingWindowFeature(data, sad_scores.sliding_window)

        speech = self._binarize.apply(speech_prob)

        speech.uri = current_file['uri']
        return speech.to_annotation(generator='string', modality='speech')

    def get_metric(self) -> DetectionErrorRate:
        """Return new instance of detection error rate metric
        If mode set on detection.
        Otherwise, it will use the self.loss function for precision and recall
        """
        if self.detection:
            return DetectionErrorRate(collar=0.0, skip_overlap=False)
        else:
            raise NotImplementedError()

    def loss(self, current_file: dict, hypothesis=None):
        reference = current_file['annotation']
        uem = get_annotated(current_file)

        precision = DetectionPrecision()
        recall = DetectionRecall()

        p = precision(reference, hypothesis, uem=uem)
        r = recall(reference, hypothesis, uem=uem)

        if p > self.precision:
            return 1. - r
        else:
            return 1. + (1. - p)
