#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

from pyannote.core import Annotation, Timeline
from pyannote.core import SlidingWindowFeature

from pyannote.audio.signal import Binarize
from pyannote.audio.features import Precomputed

from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier
from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionErrorRate


class SpeakerActivityDetection(Pipeline):
    """Single speaker activity pipeline

    Parameters
    ----------
    scores : `Path`, optional
        Path to precomputed scores on disk.
    precision : `float`, optional
        Target detection precision. Defaults to 0.8.
    use_der: 'bool', optional
        Indicates if detection error rate must be use. Default to False

    Hyper-parameters
    ----------------
    onset, offset : `float`
        Onset/offset detection thresholds
    min_duration_on, min_duration_off : `float`
        Minimum duration in either state (overlap or not)
    pad_onset, pad_offset : `float`
        Padding duration.
    """

    def __init__(self, label,
                 scores: Optional[Path] = None,
                 precision: float = 0.8,
                 use_der: bool = False):
        super().__init__()
        self.label = label
        self.scores = scores
        if self.scores is not None:
            self._precomputed = Precomputed(self.scores)
        self.precision = precision
        self.use_der = use_der

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
        """Apply single speaker activity detection

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol. May contain a
            '<self.label>_scores' key providing precomputed scores.

        Returns
        -------
        speaker_activity : `pyannote.core.Annotation`
            Single speaker activity regions.
        """

        # precomputed overlap scores
        speaker_scores = current_file.get(self.label+'_scores')
        if speaker_scores is None:
            speaker_scores = self._precomputed(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(speaker_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(speaker_scores.data) if self.log_scale_ \
               else speaker_scores.data
        # speaker speech vs (non-speech + other speakers speech)
        if data.shape[1] > 1:
            if "SPEECH" not in self._precomputed.labels and self.label == "SPEECH":
                speech_data = np.sum(data, axis=1)
                speaker_activity_prob = SlidingWindowFeature(speech_data, speaker_scores.sliding_window)
            else:
                idx = self._precomputed.labels.index(self.label)
                speaker_activity_prob = SlidingWindowFeature(data[:, idx], speaker_scores.sliding_window)
        else:
            speaker_activity_prob = SlidingWindowFeature(data, speaker_scores.sliding_window)

        speaker_activity = self._binarize.apply(speaker_activity_prob)
        speaker_activity.uri = get_unique_identifier(current_file)
        return speaker_activity.to_annotation(generator='string', modality=self.label)

    def loss(self, current_file: dict, hypothesis=None) -> float:
        """Compute (1 - recall) at target precision

        If precision < target, return 1 + (1 - precision)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Single speaker activity regions.

        Returns
        -------
        error : `float`
            1. - segment coverage.
        """
        if self.label in ["CHI", "FEM", "KCHI", "MAL"]:
            reference = current_file["annotation"].subset([self.label])
        elif self.label == "SPEECH":
            reference = current_file["annotation"]
        else:
            raise ValueError("The label you want to optimize (%s) does "
                             "not belong to [KCHI,CHI,MAL,FEM,SPEECH]" % self.label)
        uem = get_annotated(current_file)

        if not self.use_der:
            precision = DetectionPrecision()
            recall = DetectionRecall()

            p = precision(reference, hypothesis, uem=uem)
            r = recall(reference, hypothesis, uem=uem)

            if p > self.precision:
                return 1. - r
            else:
                return 1. + (1. - p)
        else:
            metric = DetectionErrorRate(collar=0.0, skip_overlap=False)
            deter = metric(reference, hypothesis, uem=uem)
            return deter
