#!/usr/bin/env python
# encoding: utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2020 CNRS
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

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
# Signal processing
"""

from typing import List, Literal

import numpy as np
import scipy.signal

from pyannote.core import Segment, SlidingWindowFeature, Timeline
from pyannote.core.utils.generators import pairwise

ThresholdScale = Literal["absolute", "relative", "percentile"]


def scale_threshold(
    thresholds: List[float],
    scores: SlidingWindowFeature = None,
    scale: ThresholdScale = "absolute",
) -> List[float]:

    """

    Parameters
    ----------
    thresholds : list of float
        Thresholds.
    scores : SlidingWindowFeature, optional
        One-dimensional scores.
    scale : {"absolute", "relative", "percentile"}, optional
        Set to "relative" to scale threshold relatively to min/max.
        Set to "percentile" to make them relative to 1% and 99% percentiles.
        Defaults to "absolute" (do not scale them)
    """

    if scale == "absolute":
        mini = 0
        maxi = 1

    elif scale == "relative":
        mini = np.nanmin(scores.data)
        maxi = np.nanmax(scores.data)

    elif scale == "percentile":
        mini = np.nanpercentile(scores.data, 1)
        maxi = np.nanpercentile(scores.data, 99)

    return [mini + t * (maxi - mini) for t in thresholds]


class Binarize:
    """Binarize detection scores using hysteresis thresholding

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to 0.5.
    scale : {"absolute", "relative", "percentile"}, optional
        Set to "relative" to make onset/offset relative to min/max.
        Set to "percentile" to make them relative 1% and 99% percentiles.
        Defaults to "absolute".
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend actiev regions by moving their end time by that many seconds.
        Defaults to 0s.

    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.
    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: float = 0.5,
        scale: ThresholdScale = "absolute",
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset
        self.scale = scale

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

    def __call__(self, scores: SlidingWindowFeature):
        """Binarize detection scores

        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.

        Returns
        -------
        active : Timeline
            Active regions.
        """

        if scores.dimension != 1:
            raise ValueError("Binarize expects one-dimensional scores.")

        onset, offset = scale_threshold(
            (self.onset, self.offset), scores=scores, scale=self.scale
        )

        num_frames = len(scores)
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # timeline meant to store 'active' regions
        active = Timeline()

        # initial state
        start = timestamps[0]
        is_active = scores[0] > self.onset

        for t, y in zip(timestamps[1:], scores[1:]):

            # currently active
            if is_active:
                # switching from active to inactive
                if y < offset:
                    region = Segment(start - self.pad_onset, t + self.pad_offset)
                    active.add(region)
                    start = t
                    is_active = False

            # currently inactive
            else:
                # switching from inactive to active
                if y > onset:
                    start = t
                    is_active = True

        # if active at the end, add final region
        if is_active:
            region = Segment(start - self.pad_onset, t + self.pad_offset)
            active.add(region)

        # because of padding, some active regions might be overlapping: merge them.
        active = active.support()

        # remove short active regions
        active = Timeline([s for s in active if s.duration > self.min_duration_on])

        # fill short inactive regions
        inactive = active.gaps()
        for s in inactive:
            if s.duration < self.min_duration_off:
                active.add(s)

        return active.support()


class Peak:
    """Peak detection

    Parameters
    ----------
    alpha : float, optional
        Peak threshold. Defaults to 0.5
    scale : {"absolute", "relative", "percentile"}, optional
        Set to "relative" to make alpha relative to min/max.
        Set to "percentile" to make it relative 1% and 99% percentiles.
        Defaults to "absolute".
    min_duration : float, optional
        Minimum elapsed time between two consecutive peaks. Defaults to 1 second.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_duration: float = 1.0,
        scale: ThresholdScale = "absolute",
    ):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.min_duration = min_duration

    def __call__(self, scores: SlidingWindowFeature):
        """Peak detection

        Parameter
        ---------
        scores : SlidingWindowFeature
            Detection scores.

        Returns
        -------
        segmentation : Timeline
            Partition.
        """

        if scores.dimension != 1:
            raise ValueError("Peak expects one-dimensional scores.")

        num_frames = len(scores)
        frames = scores.sliding_window

        precision = frames.step
        order = max(1, int(np.rint(self.min_duration / precision)))
        indices = scipy.signal.argrelmax(scores[:], order=order)[0]

        (alpha,) = scale_threshold((self.alpha,), scores=scores, scale=self.scale)

        peak_time = np.array([frames[i].middle for i in indices if scores[i] > alpha])
        boundaries = np.hstack([[frames[0].start], peak_time, [frames[num_frames].end]])

        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation
