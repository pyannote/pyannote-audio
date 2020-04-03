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
import numpy as np

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.database import get_annotated

from pyannote.audio.utils.signal import Binarize
from pyannote.audio.features import Precomputed

from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.metrics import f_measure
from pyannote.audio.features.wrapper import Wrapper, Wrappable


class OverlapDetection(Pipeline):
    """Overlap detection pipeline

    Parameters
    ----------
    scores : Wrappable, optional
        Describes how raw overlapped speech detection scores should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to "@ovl_scores" that indicates that protocol files provide
        the scores in the "ovl_scores" key.
    precision : `float`, optional
        Target detection precision. Defaults to 0.9.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing recall at
        target precision.


    Hyper-parameters
    ----------------
    onset, offset : `float`
        Onset/offset detection thresholds
    min_duration_on, min_duration_off : `float`
        Minimum duration in either state (overlap or not)
    pad_onset, pad_offset : `float`
        Padding duration.
    """

    def __init__(
        self, scores: Wrappable = None, precision: float = 0.9, fscore: bool = False
    ):
        super().__init__()

        if scores is None:
            scores = "@ovl_scores"
        self.scores = scores
        self._scores = Wrapper(self.scores)

        self.precision = precision
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
        """Apply overlap detection

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol. May contain a
            'ovl_scores' key providing precomputed scores.

        Returns
        -------
        overlap : `pyannote.core.Annotation`
            Overlap regions.
        """

        ovl_scores = self._scores(current_file)

        # if this check has not been done yet, do it once and for all
        if not hasattr(self, "log_scale_"):
            # heuristic to determine whether scores are log-scaled
            if np.nanmean(ovl_scores.data) < 0:
                self.log_scale_ = True
            else:
                self.log_scale_ = False

        data = np.exp(ovl_scores.data) if self.log_scale_ else ovl_scores.data

        # overlap vs. non-overlap
        if data.shape[1] > 1:
            overlap_prob = SlidingWindowFeature(
                1.0 - data[:, 0], ovl_scores.sliding_window
            )
        else:
            overlap_prob = SlidingWindowFeature(data, ovl_scores.sliding_window)

        overlap = self._binarize.apply(overlap_prob)

        overlap.uri = current_file.get("uri", None)
        return overlap.to_annotation(generator="string", modality="overlap")

    @staticmethod
    def to_overlap(reference: Annotation) -> Annotation:
        """Get overlapped speech reference annotation

        Parameters
        ----------
        reference : Annotation
            File yielded by pyannote.database protocols.

        Returns
        -------
        overlap : `pyannote.core.Annotation`
            Overlapped speech reference.
        """

        overlap = Timeline(uri=reference.uri)
        for (s1, t1), (s2, t2) in reference.co_iter(reference):
            l1 = reference[s1, t1]
            l2 = reference[s2, t2]
            if l1 == l2:
                continue
            overlap.add(s1 & s2)
        return overlap.support().to_annotation()

    def get_metric(self, **kwargs) -> DetectionPrecisionRecallFMeasure:
        """Get overlapped speech detection metric

        Returns
        -------
        metric : DetectionPrecisionRecallFMeasure
            Detection metric.
        """

        if not self.fscore:
            raise NotImplementedError()

        class _Metric(DetectionPrecisionRecallFMeasure):
            def compute_components(
                _self,
                reference: Annotation,
                hypothesis: Annotation,
                uem: Timeline = None,
                **kwargs
            ) -> dict:
                return super().compute_components(
                    self.to_overlap(reference), hypothesis, uem=uem, **kwargs
                )

        return _Metric()

    def loss(self, current_file: dict, hypothesis: Annotation) -> float:
        """Compute (1 - recall) at target precision

        If precision < target, return 1 + (1 - precision)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Overlap regions.

        Returns
        -------
        error : `float`
            1. - segment coverage.
        """

        precision = DetectionPrecision()
        recall = DetectionRecall()

        if "overlap_reference" in current_file:
            overlap_reference = current_file["overlap_reference"]

        else:
            reference = current_file["annotation"]
            overlap_reference = self.to_overlap(reference)
            current_file["overlap_reference"] = overlap_reference

        uem = get_annotated(current_file)
        p = precision(overlap_reference, hypothesis, uem=uem)
        r = recall(overlap_reference, hypothesis, uem=uem)

        if p > self.precision:
            return 1.0 - r
        return 1.0 + (1.0 - p)
