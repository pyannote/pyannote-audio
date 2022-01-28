# The MIT License (MIT)
#
# Copyright (c) 2022- CNRS
#
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

from functools import singledispatchmethod
from typing import Optional

import numpy as np

from pyannote.audio.utils.permutation import permutate
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
    Timeline,
)
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate


def discrete_diarization_error_rate(reference: np.ndarray, hypothesis: np.ndarray):
    """Discrete diarization error rate

    Parameters
    ----------
    reference : (num_frames, num_speakers) np.ndarray
        Discretized reference diarization.
        reference[f, s] = 1 if sth speaker is active at frame f, 0 otherwise
    hypothesis : (num_frames, num_speakers) np.ndarray
        Discretized hypothesized diarization.
       hypothesis[f, s] = 1 if sth speaker is active at frame f, 0 otherwise

    Returns
    -------
    der : float
        (false_alarm + missed_detection + confusion) / total
    components : dict
        Diarization error rate components, in number of frames.
        Keys are "false alarm", "missed detection", "confusion", and "total".
    """

    reference = reference.astype(np.half)
    hypothesis = hypothesis.astype(np.half)

    # permutate hypothesis to maximize similarity to reference
    (hypothesis,), _ = permutate(reference[np.newaxis], hypothesis)

    # total speech duration (in number of frames)
    total = 1.0 * np.sum(reference)

    # false alarm and missed detection (in number of frames)
    detection_error = np.sum(hypothesis, axis=1) - np.sum(reference, axis=1)
    false_alarm = np.maximum(0, detection_error)
    missed_detection = np.maximum(0, -detection_error)

    # speaker confusion (in number of frames)
    confusion = np.sum((hypothesis != reference) * hypothesis, axis=1) - false_alarm

    false_alarm = np.sum(false_alarm)
    missed_detection = np.sum(missed_detection)
    confusion = np.sum(confusion)

    der = (false_alarm + missed_detection + confusion) / total

    return (
        der,
        {
            "false alarm": false_alarm,
            "missed detection": missed_detection,
            "confusion": confusion,
            "total": total,
        },
    )


class DiscreteDiarizationErrorRate(BaseMetric):
    """Compute diarization error rate on discretized annotations"""

    @classmethod
    def metric_name(cls):
        return "discrete diarization error rate"

    @classmethod
    def metric_components(cls):
        return ["total", "false alarm", "missed detection", "confusion"]

    def compute_components(
        self,
        reference,
        hypothesis,
        uem: Optional[Timeline] = None,
    ):
        return self.compute_components_helper(hypothesis, reference, uem=uem)

    @singledispatchmethod
    def compute_components_helper(self, hypothesis, reference, uem=None):
        klass = hypothesis.__class__.__name__
        raise NotImplementedError(
            f"Providing hypothesis as {klass} instances is not supported."
        )

    @compute_components_helper.register
    def der_from_ndarray(
        self,
        hypothesis: np.ndarray,
        reference: np.ndarray,
        uem: Optional[Timeline] = None,
        **kwargs,
    ):

        if reference.ndim != 2:
            raise NotImplementedError(
                "Only (num_frames, num_speakers)-shaped reference is supported."
            )

        if uem is not None:
            raise ValueError("Providing a UEM is not compatible with numpy arrays.")

        ref_num_frames, ref_num_speakers = reference.shape

        if hypothesis.ndim != 2:
            raise NotImplementedError(
                "Only (num_frames, num_speakers)-shaped hypothesis is supported."
            )

        hyp_num_frames, hyp_num_speakers = hypothesis.shape

        if ref_num_frames != hyp_num_frames:
            raise ValueError(
                "reference and hypothesis must have the same number of frames."
            )

        if hyp_num_speakers > ref_num_speakers:
            reference = np.pad(
                reference, ((0, 0), (0, hyp_num_speakers - ref_num_speakers))
            )
        elif ref_num_speakers > hyp_num_speakers:
            hypothesis = np.pad(
                hypothesis, ((0, 0), (0, ref_num_speakers - hyp_num_speakers))
            )

        return discrete_diarization_error_rate(reference, hypothesis)[1]

    @compute_components_helper.register
    def der_from_swf(
        self,
        hypothesis: SlidingWindowFeature,
        reference: Annotation,
        uem: Optional[Timeline] = None,
    ):

        ndim = hypothesis.data.ndim
        if ndim < 2 or ndim > 3:
            raise NotImplementedError(
                "Only (num_frames, num_speakers) or (num_chunks, num_frames, num_speakers)-shaped "
                "hypothesis is supported."
            )

        # use hypothesis support and resolution when provided as (num_frames, num_speakers)
        if ndim == 2:
            support = hypothesis.extent
            resolution = hypothesis.sliding_window

        # use hypothesis support and estimate resolution when provided as (num_chunks, num_frames, num_speakers)
        elif ndim == 3:
            chunks = hypothesis.sliding_window
            num_chunks, num_frames, _ = hypothesis.data.shape
            support = Segment(chunks[0].start, chunks[num_chunks - 1].end)
            resolution = chunks.duration / num_frames

        # discretize reference annotation
        reference = reference.discretize(support, resolution=resolution)

        # if (num_frames, num_speakers)-shaped, compute just one DER for the whole file
        if ndim == 2:
            if uem is not None:
                components = self.init_components()
                for segment in uem:
                    h = hypothesis.crop(segment)
                    if h.size == 0:
                        continue
                    r = reference.crop(segment)
                    segment_component = self.compute_components_helper(h, r)
                    for name in self.components_:
                        components[name] += segment_component[name]
                return components
            else:
                return self.compute_components_helper(hypothesis.data, reference.data)

        # if (num_chunks, num_frames, num_speakers)-shaed, compute one DER per chunk and aggregate
        elif ndim == 3:

            components = self.init_components()
            for window, hypothesis_window in hypothesis:

                if uem is not None and not uem.covers(Timeline([window])):
                    continue

                reference_window = reference.crop(window, mode="center")

                common_num_frames = min(num_frames, reference_window.shape[0])

                window_components = self.compute_components_helper(
                    hypothesis_window[:common_num_frames],
                    reference_window[:common_num_frames],
                )

                for name in self.components_:
                    components[name] += window_components[name]

            return components

    def compute_metric(self, components):
        return (
            components["false alarm"]
            + components["missed detection"]
            + components["confusion"]
        ) / components["total"]


class SlidingDiarizationErrorRate(BaseMetric):
    def __init__(self, window: float = 10.0):
        super().__init__()
        self.window = window

    @classmethod
    def metric_name(cls):
        return "window diarization error rate"

    @classmethod
    def metric_components(cls):
        return ["total", "false alarm", "missed detection", "confusion"]

    def compute_components(
        self,
        reference,
        hypothesis,
        uem: Optional[Timeline] = None,
    ):

        if uem is None:
            raise ValueError(
                "SlidingDiarizationErrorRate expects `uem` to be provided."
            )

        der = DiarizationErrorRate()

        window = SlidingWindow(duration=self.window, step=0.5 * self.window)

        for chunk in window(uem):
            _ = der(
                reference.crop(chunk), hypothesis.crop(chunk), uem=Timeline([chunk])
            )

        return der[:]

    def compute_metric(self, components):
        return (
            components["false alarm"]
            + components["missed detection"]
            + components["confusion"]
        ) / components["total"]
