#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments

from ..generators.yaafe import YaafeMixin
from ..features.utils import get_wav_duration


class SequenceLabelingAggregation(YaafeMixin, FileBasedBatchGenerator):
    """Aggregate pre-trained sequence labeling predictions on overlapping windows

    Parameters
    ----------
    sequence_labeling : SequenceLabeling
        Pre-trained sequence labeling.
    feature_extractor : YaafeFeatureExtractor
        Yaafe feature extractor
    duration : float, optional
    step : float, optional
        Sliding window duration and step (in seconds).
        Defaults to 3 seconds window with 750ms step.

    Usage
    -----
    >>> sequence_labeling = SequenceLabeling.from_disk('architecture.yml', 'weights.h5')
    >>> feature_extractor = YaafeFeatureExtractor(...)
    >>> aggregation = SequenceLabelingAggregation(sequence_labeling, feature_extractor)
    >>> predictions = aggregation.apply('audio.wav')
    """

    def __init__(self, sequence_labeling, feature_extractor,
                 duration=3., step=0.75):

        # feature sequence
        self.feature_extractor = feature_extractor

        # sequence labeling
        self.sequence_labeling = sequence_labeling

        # overlapping window
        self.duration = duration
        self.step = step

        # initialize segments generator
        generator = SlidingSegments(duration=duration, step=step, source='wav')

        super(SequenceLabelingAggregation, self).__init__(generator, batch_size=-1)
        # TODO  setting batch_size to -1 results in one big mono_batch
        # containing every sequence from every sliding window position.
        # We should update this class to support actual batches. Note that this
        # is not at all straightforward because of how predictions are
        # accumulated through overlapping sequences

    def signature(self):
        """See `FileBasedBatchGenerator` base class for details"""
        shape = self.get_shape()
        return {'type': 'sequence', 'shape': shape}

    def postprocess_sequence(self, mono_batch):
        """

        Parameter
        ---------
        mono_batch : (n_sequences, n_samples, n_features) numpy array
            Mono-batch of sequences of features

        Returns
        -------
        prediction : (n_sequences, n_samples, n_classes) numpy array
            Mono-batch of sequences of predictions

        See `FileBasedBatchGenerator` base class for details.
        """
        return self.sequence_labeling.predict(mono_batch)

    def apply(self, wav):
        """

        Parameter
        ---------
        wav : str
            Path to wav audio file

        Returns
        -------
        predictions : SlidingWindowFeature

        """

        # apply sequence labeling to the whole file
        current_file = {'uri': wav, 'medium': {'wav': wav}}
        predictions = next(self.from_file(current_file))
        n_sequences, _, n_classes = predictions.shape

        # estimate total number of frames (over the duration of the whole file)
        # based on feature extractor internal sliding window and file duration
        samples_window = self.feature_extractor.sliding_window()
        n_samples = samples_window.samples(get_wav_duration(wav)) + 3

        # +3 is a hack to avoid later IndexError resulting from rounding error
        # when cropping samples_window

        # k[i] contains the number of sequences that overlap with frame #i
        k = np.zeros((n_samples, ), dtype=np.int8)

        # y[i] contains the sum of predictions for frame #i
        # over all overlapping samples
        y = np.zeros((n_samples, n_classes), dtype=np.float32)

        # sequence sliding window
        sequence_window = SlidingWindow(duration=self.duration, step=self.step)

        # accumulate predictions over all sequences
        for i in range(n_sequences):

            # position of sequence #i
            window = sequence_window[i]

            # indices of frames overlapped by sequence #i
            indices = samples_window.crop(
                window, mode='center', fixed=self.duration)

            # accumulate predictions
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1
            y[indices] += predictions[i, :, :]

        # average prediction
        y = (y.T / np.maximum(k, 1)).T

        # returns the whole thing as SlidingWindowFeature
        return SlidingWindowFeature(y, samples_window)
