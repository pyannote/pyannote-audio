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


class SequenceEmbedding(SequenceLabeling):
    """Sequence embedding

    Parameters
    ----------
    model : nn.Module
        Pre-trained sequence embedding model.
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
    """

    def __init__(self, model, feature_extraction, duration,
                 step=None, batch_size=32, source='audio',
                 device=None):

        super(SequenceEmbedding, self).__init__(
            model, feature_extraction, duration,
            step=step, source=source,
            batch_size=batch_size, device=device)

    @property
    def sliding_window(self):
        if self.model.internal:
            return self.feature_extractor.sliding_window()
        else:
            return SlidingWindow(duration=self.duration, step=self.step)

    def postprocess_ndarray(self, X):
        """Embed sequences

        Parameters
        ----------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        fX : numpy array
            Batch of sequence embeddings.
            (batch_size, n_samples, n_dimensions) if self.model.internal
            (batch_size, n_dimensions) if not self.model.internal
        """

        batch_size, n_samples, n_features = X.shape

        if batch_size <= self.batch_size:
            if not getattr(self.model, 'batch_first', True):
                X = np.rollaxis(X, 0, 2)
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            fX = self.model(X).to('cpu').numpy()

            if fX.ndim == 3:
                if not getattr(self.model, 'batch_first', True):
                    fX = np.rollaxis(fX, 1, 0)
            return fX

        batches = batchify(iter(X), {'type': 'ndarray'},
                           batch_size=self.batch_size,
                           incomplete=True, prefetch=0)
        return np.vstack(self.postprocess_ndarray(x) for x in batches)

    def apply(self, current_file):
        """Extract embeddings

        Can process either pyannote.database protocol items (as dict) or
        batch of precomputed feature sequences (as numpy array).

        Parameter
        ---------
        current_file : dict or numpy array
            File (from pyannote.database protocol) or batch of precomputed
            feature sequences.

        Returns
        -------
        embedding : SlidingWindowFeature or numpy array
        """

        if isinstance(current_file, np.ndarray):
            return self.postprocess_ndarray(current_file)

        if self.model.internal:
            return super(SequenceEmbedding, self).apply(current_file)

        # compute embedding on sliding window
        # over the whole duration of the file
        fX = np.vstack(
            [batch for batch in self.from_file(current_file,
                                               incomplete=True)])

        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        return SlidingWindowFeature(fX, subsequences)
