#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

"""Speech activity detection pipelines"""

from typing import Union
import numpy as np
import warnings

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.audio.utils.signal import Binarize
from pyannote.audio.features import Precomputed

from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.audio.features.wrapper import Wrapper, Wrappable


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

        speech = current_file["annotation"].get_timeline().support()
        return speech.to_annotation(generator="string", modality="speech")


class SpeechActivityDetection(Pipeline):
    """Speech activity detection pipeline

    Parameters
    ----------
    scores : Wrappable, optional
        Describes how raw speech activity detection scores should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to "@sad_scores" that indicates that protocol files provide
        the scores in the "sad_scores" key.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.

    Hyper-parameters
    ----------------
    onset, offset : `float`
        Onset/offset detection thresholds
    min_duration_on, min_duration_off : `float`
        Minimum duration in either state (speech or not)
    pad_onset, pad_offset : `float`
        Padding duration.
    """

    def __init__(self, scores: Wrappable = None, fscore: bool = False):
        super().__init__()

        if scores is None:
            scores = "@sad_scores"
        self.scores = scores
        self._scores = Wrapper(self.scores)

        self.fscore = fscore

        # hyper-parameters
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)
        self.min_duration_on = Uniform(0.0, 2.0)
        self.min_duration_off = Uniform(0.0, 2.0)
        self.pad_onset = Uniform(-1.0, 1.0)
        self.pad_offset = Uniform(-1.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
            pad_onset=self.pad_onset,
            pad_offset=self.pad_offset,
        )

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

        sad_scores = self._scores(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(sad_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(sad_scores.data) if self.log_scale_ else sad_scores.data

        # speech vs. non-speech
        if data.shape[1] > 1:
            speech_prob = SlidingWindowFeature(
                1.0 - data[:, 0], sad_scores.sliding_window
            )
        else:
            speech_prob = SlidingWindowFeature(data, sad_scores.sliding_window)

        speech = self._binarize.apply(speech_prob)

        speech.uri = current_file.get("uri", None)
        return speech.to_annotation(generator="string", modality="speech")

    def get_metric(
        self, parallel=False
    ) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(
                collar=0.0, skip_overlap=False, parallel=parallel
            )
        else:
            return DetectionErrorRate(collar=0.0, skip_overlap=False, parallel=parallel)
