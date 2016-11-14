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


from .yaafe import YaafeMixin
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingLabeledSegments


class FixedDurationSequences(YaafeMixin, FileBasedBatchGenerator):
    """(X_batch, y_batch) batch generator

    Yields batches made of sequences obtained using a sliding window over the
    coverage of the reference. Heterogeneous segments (i.e. containing more
    than one label) are skipped.

    Parameters
    ----------
    feature_extractor : YaafeFeatureExtractor
    duration: float, optional
    step: float, optional
        Duration and step of sliding window (in seconds).
        Default to 3s and 750ms.

    Returns
    -------
    X_batch : (batch_size, n_samples, n_features) numpy array
        Batch of feature sequences
    y_batch : (batch_size, ) numpy array
        Batch of corresponding labels

    Usage
    -----
    >>> batch_generator = FixedDurationSequences(feature_extractor)
    >>> for X_batch, y_batch in batch_generator.from_file(current_file):
    ...     # do something with
    """

    def __init__(self, feature_extractor, duration=3.0,
                 step=0.75, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step

        segment_generator = SlidingLabeledSegments(duration=duration,
                                                   step=step)
        super(FixedDurationSequences, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):
        return (
            {'type': 'sequence', 'shape': self.get_shape()},
            {'type': 'label'}
        )
