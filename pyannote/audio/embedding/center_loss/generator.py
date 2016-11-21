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
# Grégory GELLY


import numpy as np
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.generators.indices import random_label_index
from pyannote.audio.generators.labels import FixedDurationSequences
from pyannote.audio.generators.labels import VariableDurationSequences
from pyannote.audio.embedding.callbacks import UpdateGeneratorEmbedding


class SequenceGenerator(object):

    def __init__(self, feature_extractor, file_generator,
                 duration=3.0, min_duration=None, overlap=0.8,
                 per_label=3):

        super(SequenceGenerator, self).__init__()

        self.feature_extractor = feature_extractor
        self.file_generator = file_generator
        self.duration = duration
        self.min_duration = min_duration
        self.overlap = overlap
        self.per_label = per_label

        if self.min_duration is None:
            self.generator_ = FixedDurationSequences(
                self.feature_extractor,
                duration=self.duration,
                step=(1 - self.overlap) * self.duration,
                batch_size=-1)
        else:
            self.generator_ = VariableDurationSequences(
                self.feature_extractor,
                max_duration=self.duration,
                min_duration=self.min_duration,
                batch_size=-1)

        self.sequence_generator_ = self.iter_sequences()

        # consume first element of generator
        # this is meant to pre-generate all labeled sequences once and for all
        # and also to precompute the number of unique labels
        next(self.sequence_generator_)


    def iter_sequences(self):

        # pre-generate all labeled sequences (from the whole training set)
        # this might be huge in memory
        X, y = zip(*self.generator_(self.file_generator))
        X = np.vstack(X)
        y = np.hstack(y)

        # keep track of number of labels and rename labels to integers
        unique, y = np.unique(y, return_inverse=True)
        self.n_labels = len(unique)

        generator = random_label_index(
            y, per_label=self.per_label, return_label=False)

        # HACK (see __init__ for details on why this is done)
        yield

        while True:
            i = next(generator)
            yield X[i], y[i]

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.sequence_generator_)

    @property
    def shape(self):
        return self.generator_.shape

    def signature(self):
        shape = self.shape
        return (
            {'type': 'sequence', 'shape': shape},
            {'type': 'label'}
        )


class CenterLossBatchGenerator(BaseBatchGenerator):
    """Triplet generator for triplet loss sequence embedding

    Generates ([X], derivatives) batch tuples where
      * X are sequences
      * derivatives are ...

    Parameters
    ----------
    feature_extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC instance)
    file_generator: iterable
        File generator (the training set, typically)
    compute_derivatives: callable
        ...
    distance: {'sqeuclidean', 'cosine', 'angular'}
        Distance for which the embedding is optimized. Defaults to 'angular'.
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    min_duration: float, optional
        Sequence minimum duration. When provided, generates sequences with
        random duration in range [min_duration, duration]. Defaults to
        fixed-duration sequences.
    per_label: int, optional
        Number of samples per label. Defaults to 3.
    per_fold: int, optional
        Number of labels per fold. Defaults to 20.
    per_batch: int, optional
        Number of folds per batch. Defaults to 12.
    n_threads: int, optional
        Defaults to 6.
    """

    def __init__(self, feature_extractor, file_generator, compute_derivatives,
                 distance='angular', duration=3.0, min_duration=None,
                 per_label=3, per_fold=20, per_batch=12, n_threads=1):

        self.sequence_generator_ = SequenceGenerator(
            feature_extractor, file_generator,
             duration=duration, min_duration=min_duration,
             per_label=per_label)

        self.n_labels = self.sequence_generator_.n_labels
        self.per_label = per_label
        self.per_fold = per_fold
        self.per_batch = per_batch
        self.n_threads = n_threads

        batch_size = self.per_label * self.per_fold * self.per_batch
        super(CenterLossBatchGenerator, self).__init__(
            self.sequence_generator_, batch_size=batch_size)

        self.compute_derivatives = compute_derivatives

    # def signature(self):
    #     return self.sequence_generator_.signature()

    @property
    def shape(self):
        return self.sequence_generator_.shape

    def get_samples_per_epoch(self, protocol, subset='train'):
        """
        Parameters
        ----------
        protocol : pyannote.database.protocol.protocol.Protocol
        subset : {'train', 'development', 'test'}, optional

        Returns
        -------
        samples_per_epoch : int
            Number of samples per epoch.
        """
        n_folds = self.n_labels / self.per_fold + 1
        return self.batch_size * n_folds

    # this callback will make sure the internal embedding is always up to date
    def callbacks(self, extract_embedding=None):
        callback = UpdateGeneratorEmbedding(
            self, extract_embedding=extract_embedding, name='embedding')
        return [callback]

    def postprocess(self, batch):

        sequences, labels = batch

        embeddings = self.embedding.transform(
            sequences, batch_size=self.per_fold * self.per_label)

        [costs, derivatives] = self.compute_derivatives(embeddings, labels)

        return sequences, derivatives
