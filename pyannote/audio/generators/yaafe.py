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


class YaafeMixin:

    def get_shape(self):
        return self.yaafe_get_shape()

    def yaafe_get_shape(self):
        n_samples = self.feature_extractor.sliding_window().samples(
            self.duration, mode='center')
        dimension = self.feature_extractor.dimension()
        return (n_samples, dimension)

    # defaults to features pre-computing
    def preprocess(self, current_file, identifier=None):
        return self.yaafe_preprocess(
            current_file, identifier=identifier)

    def yaafe_preprocess(self, current_file, identifier=None):

        if not hasattr(self, 'preprocessed_'):
            self.preprocessed_ = {}

        if identifier in self.preprocessed_.setdefault('X', {}):
            return current_file

        wav = current_file['medium']['wav']
        features = self.feature_extractor(wav)

        self.preprocessed_['X'][identifier] = features

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        return self.yaafe_process_segment(
            segment, signature=signature, identifier=identifier)

    def yaafe_process_segment(self, segment, signature=None, identifier=None):
        duration = signature.get('duration', None)
        return self.preprocessed_['X'][identifier].crop(
            segment, mode='center', fixed=duration)
