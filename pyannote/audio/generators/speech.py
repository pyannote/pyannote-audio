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
from pyannote.core import SlidingWindowFeature
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from scipy.stats import zscore
import numpy as np


class SpeechActivityDetectionBatchGenerator(YaafeMixin,
                                            FileBasedBatchGenerator):

    def __init__(self, feature_extractor,
                 duration=3.2, step=0.8, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step

        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source='annotated')
        super(SpeechActivityDetectionBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):

        shape = self.yaafe_get_shape()
        dimension = 2

        return [
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': (shape[0], dimension)}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""

        current_file = self.yaafe_preprocess(
            current_file, identifier=identifier)

        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()

        y = np.zeros((n_samples + 1, 2), dtype=np.int8)
        # [0,1] ==> speech / [1, 0] ==> non speech / [0, 0] ==> unknown

        annotated = current_file['annotated']
        annotation = current_file['annotation']

        coverage = annotation.get_timeline().coverage()

        for gap in coverage.gaps(annotated):
            indices = sw.crop(gap, mode='loose')
            y[indices, 0] = 1

        for segment in coverage:
            indices = sw.crop(segment, mode='loose')
            y[indices, 1] = 1

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.yaafe_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)

        return [X, y]


class OverlappingSpeechDetectionBatchGenerator(YaafeMixin,
                                               FileBasedBatchGenerator):

    def __init__(self, feature_extractor, duration=3.2,
                 step=0.8, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step

        # source = 'coverage' ensures only speech regions are covered
        segment_generator = SlidingSegments(duration=duration,
                                            step=step,
                                            source='coverage')
        super(OverlappingSpeechDetectionBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):

        shape = self.yaafe_get_shape()
        dimension = 2

        return [
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': (shape[0], dimension)}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""

        current_file = self.yaafe_preprocess(
            current_file, identifier=identifier)

        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()

        y = np.zeros((n_samples + 1, 2), dtype=np.int8)
        # [0,1] ==> overlapping speech / [1, 0] ==> speech / [0, 0] ==> unknown

        annotated = current_file['annotated']
        annotation = current_file['annotation']

        timeline = annotation.get_timeline()

        # speech regions
        for segment in timeline:
            indices = sw.crop(segment, mode='loose')
            y[indices, 0] = 1

        # overlapping speech regions
        for segment, other_segment in timeline.co_iter(timeline):
            if segment == other_segment:
                continue

            overlap = segment & other_segment
            indices = sw.crop(overlap, mode='loose')
            y[indices, 1] = 1
            y[indices, 0] = 0

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.yaafe_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)

        return [X, y]
