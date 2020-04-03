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

from typing import Optional
from typing import Union
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
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure

from pyannote.audio.features.wrapper import Wrapper, Wrappable


class SpeakerChangeDetection(Pipeline):
    """Speaker change detection pipeline

    Parameters
    ----------
    scores : Wrappable, optional
        Describes how raw speaker change detection scores should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to "@scd_scores" that indicates that protocol files provide
        the scores in the "scd_scores" key.
    purity : `float`, optional
        Target segments purity. Defaults to 0.95.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing coverage at
        given target `purity`.
    diarization : bool, optional
        Use diarization purity and coverage. Defaults to segmentation purity
        and coverage.

    Hyper-parameters
    ----------------
    alpha : `float`
        Peak detection threshold.
    min_duration : `float`
        Segment minimum duration.
    """

    def __init__(
        self,
        scores: Wrappable = None,
        purity: Optional[float] = 0.95,
        fscore: bool = False,
        diarization: bool = False,
    ):
        super().__init__()

        if scores is None:
            scores = "@scd_scores"
        self.scores = scores
        self._scores = Wrapper(self.scores)

        self.purity = purity
        self.fscore = fscore
        self.diarization = diarization

        # hyper-parameters
        self.alpha = Uniform(0.0, 1.0)
        self.min_duration = Uniform(0.0, 10.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._peak = Peak(alpha=self.alpha, min_duration=self.min_duration)

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

        scd_scores = self._scores(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(scd_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(scd_scores.data) if self.log_scale_ else scd_scores.data

        # take the final dimension
        # (in order to support both classification, multi-class classification,
        # and regression scores)
        change_prob = SlidingWindowFeature(data[:, -1], scd_scores.sliding_window)

        # peak detection
        change = self._peak.apply(change_prob)
        change.uri = current_file.get("uri", None)

        return change.to_annotation(generator="string", modality="audio")

    def get_metric(
        self, parallel=False
    ) -> Union[DiarizationPurityCoverageFMeasure, SegmentationPurityCoverageFMeasure]:
        """Return new instance of f-score metric"""

        if not self.fscore:
            raise NotImplementedError()

        if self.diarization:
            return DiarizationPurityCoverageFMeasure(parallel=parallel)

        return SegmentationPurityCoverageFMeasure(tolerance=0.5, parallel=parallel)

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
        reference = current_file["annotation"]
        uem = get_annotated(current_file)
        f_measure = metric(reference, hypothesis, uem=uem)
        purity, coverage, _ = metric.compute_metrics()
        if purity > self.purity:
            return 1.0 - coverage
        else:
            return 1.0 + (1.0 - purity)
