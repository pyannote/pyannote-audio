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

from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.audio.utils.signal import Peak
from pyannote.audio.features import Precomputed

from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure


class SpeakerChangeDetection(Pipeline):
    """Speaker change detection pipeline

    Parameters
    ----------
    scores : `Path`, optional
        Path to precomputed scores on disk.
    purity : `float`, optional
        Target segments purity. Defaults to 0.95.

    Hyper-parameters
    ----------------
    alpha : `float`
        Peak detection threshold.
    min_duration : `float`
        Segment minimum duration.
    """

    def __init__(self, scores: Optional[Path] = None,
                       purity: Optional[float] = 0.95):
        super().__init__()

        self.scores = scores
        if self.scores is not None:
            self._precomputed = Precomputed(self.scores)
        self.purity = purity

        # hyper-parameters
        self.alpha = Uniform(0., 1.)
        self.min_duration = Uniform(0., 10.)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._peak = Peak(alpha=self.alpha,
                          min_duration=self.min_duration)

    def __call__(self, current_file: dict) -> Annotation:
        """Apply change detection

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.  May contain a
            'scd_scores' key providing precomputed scores.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Speech regions.
        """

        # precomputed SCD scores
        scd_scores = current_file.get('scd_scores')
        if scd_scores is None:
            scd_scores = self._precomputed(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(scd_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(scd_scores.data) if self.log_scale_ \
               else scd_scores.data

        # take the final dimension
        # (in order to support both classification, multi-class classification,
        # and regression scores)
        change_prob = SlidingWindowFeature(
            data[:, -1],
            scd_scores.sliding_window)

        # peak detection
        change = self._peak.apply(change_prob)
        change.uri = current_file['uri']

        return change.to_annotation(generator='string', modality='audio')

    def loss(self, current_file: dict, hypothesis: Annotation) -> float:
        """Compute (1 - coverage) at target purity

        If purity < target, return 1 + (1 - purity)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Speech regions.

        Returns
        -------
        error : `float`
            1. - segment coverage.
        """

        metric = SegmentationPurityCoverageFMeasure(tolerance=0.500, beta=1)
        reference  = current_file['annotation']
        uem = get_annotated(current_file)
        f_measure = metric(reference, hypothesis, uem=uem)
        purity, coverage, _ = metric.compute_metrics()
        if purity > self.purity:
            return 1. - coverage
        else:
            return 1. + (1. - purity)
