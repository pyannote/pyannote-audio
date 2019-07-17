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

import torch
import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.generators.batch import batchify
import torch.nn as nn


class SequenceEmbedding(SequenceLabeling):
    """Sequence embedding

    Parameters
    ----------
    model : `nn.Module` or `str`
        Model (or path to model). When a path, the directory structure created
        by pyannote-speaker-embedding should be kept unchanged so that one can
        find the corresponding configuration file automatically.
    feature_extraction : callable, optional
        Feature extractor. When not provided and `model` is a path, it is
        inferred directly from the configuration file.
    duration : float, optional
        Subsequence duration, in seconds. When `model` is a path and `duration`
        is not provided, it is inferred directly from the configuration file.
    min_duration : float, optional
        When provided, will do its best to yield segments of length `duration`,
        but shortest segments are also permitted (as long as they are longer
        than `min_duration`). When `model` is a path and `min_duration` is not
        provided, it is inferred directly from the configuration file.
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    batch_size : int, optional
        Defaults to 32.
    device : `torch.device` or `str`, optional
        Defaults to CPU.
    """

    def __init__(self, model=None, feature_extraction=None,
                 step=None, duration=None, min_duration=None,
                 batch_size=32, device=None):

        # support for providing device as 'cpu' or 'cuda'
        if isinstance(device, str):
            device = torch.device(device)

        if not isinstance(model, nn.Module):

            from pyannote.audio.applications.speaker_embedding \
                import SpeakerEmbedding

            # training = False is important to ensure no data augmentation
            # is added (as it would likely degrade performance)
            app = SpeakerEmbedding.from_model_pt(model, training=False)

            model = app.model_

            if feature_extraction is None:
                feature_extraction = app.feature_extraction_

            if duration is None:
                duration = app.task_.duration

            if duration is None:
                duration = app.task_.max_duration
                if min_duration is None:
                    min_duration = app.task_.min_duration

        super().__init__(model=model, feature_extraction=feature_extraction,
                         step=step, duration=duration, min_duration=min_duration,
                         batch_size=batch_size, device=device)

    @property
    def dimension(self):
        """Dimension of embeddings"""
        return self.model.dimension

    @property
    def sliding_window(self):
        return SlidingWindow(duration=self.duration, step=self.step)

    def apply(self, X):
        """Embed (fixed-length) sequences

        Parameters
        ----------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        fX : (batch_size, n_dimensions) numpy array
            Batch of sequence embeddings.
        """

        batch_size, n_samples, n_features = X.shape

        # this test is needed because .apply() may be called
        # with a ndarray of arbitrary size as input
        if batch_size <= self.batch_size:
            return self.forward(X)

        # if X contains too large a batch, split it in smaller batches...
        batches = batchify(iter(X), {'@': (None, np.stack)},
                           batch_size=self.batch_size,
                           incomplete=True, prefetch=0)

        # ... and process them in order, before re-concatenating them
        return np.vstack([self.apply(x) for x in batches])

    def __call__(self, current_file):
        """Extract embeddings on a sliding window

        Parameters
        ----------
        current_file : `dict`
            File (from pyannote.database protocol)

        Returns
        -------
        embeddings : `SlidingWindowFeature`
            Extracted embeddings
        """

        # compute embedding on sliding window
        # over the whole duration of the source
        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]

        if not batches:
            fX = np.zeros((0, self.dimension))
        else:
            fX = np.vstack(batches)

        subsequences = SlidingWindow(duration=self.duration,
                                     step=self.step)
        return SlidingWindowFeature(fX, subsequences)

    def get_context_duration(self):
        """

        Returns
        -------
        context : float
            Context duration, in seconds.
        """
        return self.feature_extraction.get_context_duration()

    def crop(self, current_file, segment):
        """Extract embeddings from a specific time range

        Parameters
        ----------
        current_file : `dict`
            File (from pyannote.database protocol)
        segment : `Segment` or `Timeline`, optional
            Time range from which to extract embeddings.

        Returns
        -------
        embeddings : (n_windows, dimension) `numpy array`
            Extracted embeddings
        """

        # HACK: change internal SlidingSegment's source to only extract
        # embeddings on provided "crop". keep track of original source
        # to set it back before the function returns
        source = self.generator.source
        self.generator.source = segment

        # compute embedding on sliding window
        # over the whole duration of the source
        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]

        self.generator.source = source

        if not batches:
            return np.zeros((0, self.dimension))

        return np.vstack(batches)
