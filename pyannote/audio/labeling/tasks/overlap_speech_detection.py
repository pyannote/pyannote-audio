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


from .base import LabelingTask
from .speaker_change_detection import SpeakerChangeDetectionGenerator


class OverlapSpeechDetectionGenerator(SpeakerChangeDetectionGenerator):
    """Generate batch of segments with associated frame-wise labels

    Parameters
    ----------
    precomputed : pyannote.audio.features.Precomputed
        Precomputed features
    duration : float, optional
        Use fixed duration segments with this `duration`.
    batch_size : int, optional
        Batch size. Defaults to 32.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def fill_y(self, y, sequence, current_file):

        n_samples = len(y)
        sw = self.precomputed.sliding_window()
        left, _ = sw.crop(sequence, mode='center', return_ranges=True)[0]

        turns = current_file['annotation']
        turns = turns.crop(sequence)

        for speech in turns.get_timeline().support():
            l, r = sw.crop(speech, mode='center', return_ranges=True)[0]
            l = max(0, l - left)
            r = min(r - left, n_samples)
            y[l:r] = 1

        for (turn1, label1), (turn2, label2) in turns.co_iter(turns):
            if label1 == label2:
                continue
            l, r = sw.crop(turn1 & turn2, mode='center', return_ranges=True)[0]
            l = max(0, l - left)
            r = min(r - left, n_samples)
            y[l:r] = 2

        return

class OverlapSpeechDetection(LabelingTask):

    def __init__(self, duration=3.2, batch_size=32, parallel=1):
        super(OverlapSpeechDetection, self).__init__(duration=duration,
                                                      batch_size=batch_size,
                                                      parallel=parallel)

    def get_batch_generator(self, precomputed):
        return OverlapSpeechDetectionGenerator(
            precomputed, duration=self.duration,
            batch_size=self.batch_size, parallel=self.parallel)

    @property
    def n_classes(self):
        return 3
