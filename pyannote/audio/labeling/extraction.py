#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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
from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.database import get_unique_identifier
import torch


class SequenceLabeling(PeriodicFeaturesMixin, FileBasedBatchGenerator):
    """Sequence labeling

    Parameters
    ----------
    model : nn.Module
        Pre-trained sequence labeling model.
    feature_extraction : callable
        Feature extractor
    duration : float
        Subsequence duration, in seconds.
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    batch_size : int, optional
        Defaults to 32.
    device : torch.device, optional
        Defaults to CPU.

    Usage
    -----
    >>> model = keras.models.load_model(...)
    >>> feature_extraction = YaafeMFCC(...)
    >>> sequence_labeling = SequenceLabeling(model, feature_extraction, duration)
    >>> sequence_labeling.apply(current_file)
    """

    def __init__(self, model, feature_extraction, duration,
                 step=None, batch_size=32, source='audio', device=None):

        self.feature_extractor = feature_extraction
        self.duration = duration
        self.batch_size = batch_size
        self.device = torch.device('cpu') if device is None else device
        self.model = model.to(self.device)

        generator = SlidingSegments(duration=duration, step=step, source=source)
        self.step = generator.step if step is None else step

        super(SequenceLabeling, self).__init__(
            generator, batch_size=self.batch_size)

    @property
    def dimension(self):
        if hasattr(self.model, 'n_classes'):
            return self.model.n_classes
        elif hasattr(self.model, 'output_dim'):
            return self.model.output_dim
        else:
            raise ValueError('Model has no n_classes nor output_dim attribute.')

    @property
    def sliding_window(self):
        return self.feature_extractor.sliding_window()

    def signature(self):
        shape = self.shape
        return {'type': 'ndarray', 'shape': self.shape}

    def postprocess_ndarray(self, X):
        """Label sequences

        Parameter
        ---------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        prediction : (batch_size, n_samples, dimension) numpy array
            Batch of sequence labelings.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        prediction = self.model(X).data.to('cpu').numpy()
        return prediction

    def apply(self, current_file):
        """Compute predictions on a sliding window

        Parameter
        ---------
        current_file : dict

        Returns
        -------
        predictions : SlidingWindowFeature
        """

        # frame and sub-sequence sliding windows
        frames = self.feature_extractor.sliding_window()

        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]
        if not batches:
            data = np.zeros((0, self.dimension), dtype=np.float32)
            return SlidingWindowFeature(data, frames)

        fX = np.vstack(batches)

        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        # get total number of frames
        identifier = get_unique_identifier(current_file)
        n_frames = self.preprocessed_['X'][identifier].data.shape[0]

        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, self.dimension), dtype=np.float32)

        # k[i] is the number of sequences that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for subsequence, fX_ in zip(subsequences, fX):

            # indices of frames overlapped by subsequence
            indices = frames.crop(subsequence,
                                  mode='center',
                                  fixed=self.duration)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        return SlidingWindowFeature(data, frames)
