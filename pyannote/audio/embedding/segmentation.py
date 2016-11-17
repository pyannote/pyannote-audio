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
from pyannote.generators.fragment import TwinSlidingSegments
from ..generators.yaafe import YaafeMixin


class Segmentation(YaafeMixin, FileBasedBatchGenerator):
    """Segmentation based on sequence embedding

    Computes the euclidean distance between the embeddings of two
    (left and right) sliding windows.

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

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml', 'weights.h5')
    >>> feature_extractor = YaafeFeatureExtractor(...)
    >>> segmentation = Segmentation(sequence_embedding, feature_extractor)
    >>> predictions = segmentation.apply('audio.wav')
    >>> segmentation = Peak().apply(predictions)

    See also
    --------
    pyannote.audio.embedding.models.SequenceEmbedding
    pyannote.audio.signal.Peak

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 duration=1.000, step=0.100):

        # feature sequence
        self.feature_extractor = feature_extractor

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # (left vs. right) sliding windows
        self.duration = duration
        self.step = step
        generator = TwinSlidingSegments(duration=duration, step=step)

        super(Segmentation, self).__init__(generator, batch_size=-1)

    def signature(self):
        shape = self.shape
        return (
            {'type': 'timestamp'},
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': shape}
        )

    def postprocess_sequence(self, mono_batch):
        return self.sequence_embedding.transform(mono_batch)

    def apply(self, wav):
        """Computes distance between sliding windows embeddings

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

        t, left, right = next(self.from_file(current_file))
        y = np.sqrt(np.sum((left - right) ** 2, axis=-1))


        window = SlidingWindow(duration=2 * self.duration,
                               step=self.step, start=0.)
        return SlidingWindowFeature(y, window)
