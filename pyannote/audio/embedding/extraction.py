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

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml',
    ...                                                  'weights.h5')
    >>> feature_extraction = YaafeFeatureExtractor(...)
    >>> extraction = Extraction(sequence_embedding, feature_extraction)
    >>> embedding = extraction.apply(current_file)

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 duration=1.000, step=None, internal=None):

        # feature sequence
        self.feature_extractor = feature_extractor

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # sliding window
        self.duration = duration
        generator = SlidingSegments(duration=duration, step=step, source='wav')
        self.step = generator.step if step is None else step

        self.internal = internal

        super(Extraction, self).__init__(generator, batch_size=32)

    @property
    def dimension(self):
        internal = -1 if self.internal is None else self.internal
        return self.sequence_embedding.embedding_.layers[internal].output_shape[1:]

    @property
    def sliding_window(self):
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
        window = SlidingWindow(duration=self.duration, step=self.step, start=0.)
        batches = [batch for batch in self.from_file(current_file)]
        return SlidingWindowFeature(np.vstack(batches), window)
