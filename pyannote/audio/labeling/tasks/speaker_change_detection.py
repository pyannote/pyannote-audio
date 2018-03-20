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
from .base import LabelingTask
from pyannote.core import Segment
from .speech_activity_detection import SpeechActivityDetectionGenerator


class SpeakerChangeDetectionGenerator(SpeechActivityDetectionGenerator):
    """Generate batch of segments with associated frame-wise labels

    Parameters
    ----------
    precomputed : pyannote.audio.features.Precomputed
        Precomputed features
    duration : float, optional
        Use fixed duration segments with this `duration`.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    balance: float, optional
        Artificially increase the number of positive labels by
        labeling as positive every frame in the direct neighborhood
        (less than `balance` seconds apart) of each change point.
        Defaults to 0.1 (= 100 ms).
    batch_size : int, optional
        Batch size. Defaults to 32.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, precomputed, duration=3.2, per_epoch=3600,
                 balance=0.1, batch_size=32, parallel=1):

        super(SpeakerChangeDetectionGenerator, self).__init__(
            precomputed, duration=duration, batch_size=batch_size,
            per_epoch=per_epoch, parallel=parallel)
        self.balance = balance

    def fill_y(self, y, sequence, current_file):

        n_samples = len(y)
        sw = self.precomputed.sliding_window()
        left, _ = sw.crop(sequence, mode='center', return_ranges=True)[0]

        # extend sequence on both sides to **not** miss any speaker change
        x_sequence = Segment(sequence.start - self.balance,
                            sequence.end + self.balance)
        x_turns = current_file['annotation']
        x_turns = x_turns.crop(x_sequence).get_timeline()

        for segment in x_turns:
            for boundary in segment:
                window = Segment(boundary - self.balance,
                                 boundary + self.balance)
                l, r = sw.crop(window, mode='center', return_ranges=True)[0]
                l = max(0, l - left)
                r = min(r - left, n_samples)
                y[l:r] = 1

        return

class SpeakerChangeDetection(LabelingTask):
    """

    Parameters
    ----------
    duration : float, optional
        Use fixed duration segments with this `duration`.
    balance: float, optional
        Artificially increase the number of positive labels by
        labeling as positive every frame in the direct neighborhood
        (less than `balance` seconds apart) of each change point.
        Defaults to 0.1 (= 100 ms).
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, duration=3.2, balance=0.1, batch_size=32,
                 per_epoch=3600, parallel=1):
        super(SpeakerChangeDetection, self).__init__(
            duration=duration, batch_size=batch_size,
            per_epoch=per_epoch, parallel=parallel)
        self.balance = balance

    def get_batch_generator(self, precomputed):
        """
        Parameters
        ----------
        precomputed : pyannote.audio.features.Precomputed
            Precomputed features
        """
        return SpeakerChangeDetectionGenerator(
            precomputed, balance=self.balance, duration=self.duration,
            batch_size=self.batch_size, per_epoch=self.per_epoch,
            parallel=self.parallel)

    @property
    def n_classes(self):
        return 2
