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
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.generators.batch import batchify
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceEmbedding(SequenceLabeling):
    """Sequence embedding

    Parameters
    ----------
    model : nn.Module
        Pre-trained sequence embedding model.
    feature_extraction : callable
        Feature extractor
    duration : float, optional
        Subsequence duration, in seconds. Defaults to 1s.
    min_duration : float, optional
        When provided, will do its best to yield segments of length `duration`,
        but shortest segments are also permitted (as long as they are longer
        than `min_duration`).
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    batch_size : int, optional
        Defaults to 32.
    device : torch.device, optional
        Defaults to CPU.
    """

    def __init__(self, model, feature_extraction, duration=1,
                 min_duration=None, step=None, batch_size=32, source='audio',
                 device=None):

        super(SequenceEmbedding, self).__init__(
            model, feature_extraction, duration=duration,
            min_duration=min_duration, step=step, source=source,
            batch_size=batch_size, device=device)

    @property
    def sliding_window(self):
        return SlidingWindow(duration=self.duration, step=self.step)

    def postprocess_sequence(self, X):
        """Embed (variable-length) sequences

        Parameters
        ----------
        X : list
            List of input sequences

        Returns
        -------
        fX : numpy array
            Batch of sequence embeddings.
        """

        lengths = torch.tensor([len(x) for x in X])
        sorted_lengths, sort = torch.sort(lengths, descending=True)
        _, unsort = torch.sort(sort)

        sequences = [torch.tensor(X[i],
                                  dtype=torch.float32,
                                  device=self.device) for i in sort]
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        packed = pack_padded_sequence(padded, sorted_lengths,
                                      batch_first=True)

        cpu = torch.device('cpu')
        fX = self.model(packed).detach().to(cpu).numpy()
        return fX[unsort]

    def postprocess_ndarray(self, X):
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
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            cpu = torch.device('cpu')
            return self.model(X).detach().to(cpu).numpy()

        # if X contains too large a batch, split it in smaller batches...
        batches = batchify(iter(X), {'@': (None, np.stack)},
                           batch_size=self.batch_size,
                           incomplete=True, prefetch=0)

        # ... and process them in order, before re-concatenating them
        return np.vstack([self.postprocess_ndarray(x) for x in batches])

    def apply(self, current_file, crop=None):
        """Extract embeddings

        Can process either pyannote.database protocol items (as dict) or
        batch of precomputed feature sequences (as numpy array).

        Parameters
        ----------
        current_file : dict or numpy array
            File (from pyannote.database protocol) or batch of precomputed
            feature sequences.
        crop : Segment or Timeline, optional
            When provided, only extract corresponding embeddings.

        Returns
        -------
        embedding : SlidingWindowFeature or numpy array
        """

        # if current_file is in fact a batch of feature sequences
        # use postprocess_ndarray directly.
        if isinstance(current_file, np.ndarray):
            return self.postprocess_ndarray(current_file)

        # HACK: change internal SlidingSegment's source to only extract
        # embeddings on provided "crop". keep track of original source
        # to set it back before the function returns
        source = self.generator.source
        if crop is not None:
            self.generator.source = crop

        # compute embedding on sliding window
        # over the whole duration of the source
        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]

        self.generator.source = source

        if not batches:
            fX = np.zeros((0, self.dimension))
        else:
            fX = np.vstack(batches)

        if crop is not None:
            return fX

        subsequences = SlidingWindow(duration=self.duration,
                                     step=self.step)
        return SlidingWindowFeature(fX, subsequences)
