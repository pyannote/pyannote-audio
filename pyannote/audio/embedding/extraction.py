#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

import warnings
import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments
from pyannote.audio.generators.periodic import PeriodicFeaturesMixin


class Extraction(PeriodicFeaturesMixin, FileBasedBatchGenerator):
    """Embedding extraction

    Parameters
    ----------
    sequence_embedding : SequenceEmbedding
        Pre-trained sequence embedding.
    feature_extractor : YaafeFeatureExtractor
        Yaafe feature extractor
    duration : float, optional
    step : float, optional
        Sliding window duration and step (in seconds).
        Defaults to 5s window with 50% step.
    internal : int, optional
        Index of layer for which to return the activation.
        Defaults (-1) to returning the activation of the final layer.
    aggregate : bool, optional

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml',
    ...                                                  'weights.h5')
    >>> feature_extraction = YaafeFeatureExtractor(...)
    >>> extraction = Extraction(sequence_embedding, feature_extraction)
    >>> embedding = extraction.apply(current_file)

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 duration=1.000, step=None, internal=None, aggregate=False):

        # feature sequence
        self.feature_extractor = feature_extractor

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # sliding window
        self.duration = duration
        generator = SlidingSegments(duration=duration, step=step, source='wav')
        self.step = generator.step if step is None else step

        if aggregate and (internal is None or internal == -1):
            warnings.warn(
                '"aggregate" parameter has no effect when '
                'the output of the final layer is returned.')

        self.internal = internal
        self.aggregate = aggregate

        super(Extraction, self).__init__(generator, batch_size=32)

    @property
    def dimension(self):
        internal = -1 if self.internal is None else self.internal
        if self.aggregate:
            return self.sequence_embedding.embedding_.layers[internal].output_shape[-1]
        else:
            return self.sequence_embedding.embedding_.layers[internal].output_shape[1:]

    @property
    def sliding_window(self):
        if self.aggregate:
            return self.feature_extractor.sliding_window()
        else:
            return SlidingWindow(start=0., duration=self.duration, step=self.step)

    def signature(self):
        shape = self.shape
        return {'type': 'sequence', 'shape': shape}

    def postprocess_sequence(self, batch):
        return self.sequence_embedding.transform(
            batch, internal=self.internal)

    def apply(self, current_file):
        """Compute embeddings on a sliding window

        Parameter
        ---------
        current_file : dict
        Returns
        -------
        embeddings : SlidingWindowFeature
        """

        embeddings = np.vstack([batch for batch in self.from_file(
            current_file, incomplete=True)])

        window = SlidingWindow(duration=self.duration, step=self.step)

        if not self.aggregate:
            return SlidingWindowFeature(embeddings, window)

        # estimate total number of frames based on number of batches
        samples_window = self.feature_extractor.sliding_window()
        n_sequences, _, dimension = embeddings.shape
        duration = window[n_sequences - 1].end
        n_samples = samples_window.samples(duration) + 4

        # k[i] contains the number of sequences that overlap with frame #i
        k = np.zeros((n_samples, 1), dtype=np.int8)

        # fX[i] contains the sum of embeddings for frame #i
        # over all overlapping samples
        fX = np.zeros((n_samples, dimension), dtype=np.float32)

        for i, embedding in enumerate(embeddings):

            # indices of frames overlapped by sequence #i
            indices = samples_window.crop(window[i], mode='center',
                                          fixed=self.duration)

            fX[indices] += embeddings[i]

        fX = fX / np.maximum(k, 1)

        return SlidingWindowFeature(fX, samples_window)
