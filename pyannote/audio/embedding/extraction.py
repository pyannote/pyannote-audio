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

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments
from ..generators.yaafe import YaafeMixin


class Extraction(YaafeMixin, FileBasedBatchGenerator):
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
        Defaults to 1 second window with 100ms step.
    layer_index : int, optional
        Index of layer for which to return the activation.
        Defaults to returning the activation of the final layer.

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml', 'weights.h5')
    >>> feature_extraction = YaafeFeatureExtractor(...)
    >>> extraction = Extraction(sequence_embedding, feature_extraction)
    >>> embedding = extraction.apply('audio.wav')

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 normalize=False, duration=1.000,
                 step=0.100, layer_index=None):

        # feature sequence
        self.feature_extractor = feature_extractor
        self.normalize = normalize

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # sliding window
        self.duration = duration
        self.step = step
        generator = SlidingSegments(duration=duration, step=step, source='wav')

        self.layer_index = layer_index

        super(Extraction, self).__init__(generator, batch_size=-1)

    def signature(self):
        shape = self.get_shape()
        return {'type': 'sequence', 'shape': shape}

    def postprocess_sequence(self, mono_batch):
        return self.sequence_embedding.transform(
            mono_batch, layer_index=self.layer_index)

    def apply(self, wav):
        """Compute embeddings on a sliding window

        Parameter
        ---------
        wav : str
            Path to wav audio file

        Returns
        -------
        embeddings : SlidingWindowFeature
        """

        current_file = {'uri': wav, 'medium': {'wav': wav}}
        data = next(self.from_file(current_file))
        window = SlidingWindow(duration=self.duration,
                               step=self.step, start=0.)
        return SlidingWindowFeature(data, window)
