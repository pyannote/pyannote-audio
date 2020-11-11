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

from typing import Literal, Tuple

import numpy as np

from pyannote.core import Segment, SlidingWindowFeature, Timeline


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
        scale: Literal["absolute", "relative", "percentile"] = "absolute",
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

    def relative_thresholds(self, scores: SlidingWindowFeature) -> Tuple[float, float]:

        if self.scale == "absolute":
            mini = 0
            maxi = 1

        elif self.scale == "relative":
            mini = np.nanmin(scores.data)
            maxi = np.nanmax(scores.data)

        elif self.scale == "percentile":
            mini = np.nanpercentile(scores.data, 1)
            maxi = np.nanpercentile(scores.data, 99)

        onset = mini + self.onset * (maxi - mini)
        offset = mini + self.offset * (maxi - mini)

        return onset, offset

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
            raise ValueError()

        onset, offset = self.relative_thresholds(scores)

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
