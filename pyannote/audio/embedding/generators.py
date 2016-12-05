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


import h5py
import os.path
import numpy as np
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.generators.indices import random_label_index
from pyannote.audio.generators.labels import FixedDurationSequences
from pyannote.audio.generators.labels import VariableDurationSequences
from pyannote.audio.embedding.callbacks import UpdateGeneratorEmbedding


class SequenceGenerator(object):

    def __init__(self, feature_extractor, file_generator,
                 duration=3.0, min_duration=None, overlap=0.8,
                 per_label=3, cache=None):

        super(SequenceGenerator, self).__init__()

        self.feature_extractor = feature_extractor
        self.file_generator = file_generator
        self.duration = duration
        self.min_duration = min_duration
        self.overlap = overlap
        self.per_label = per_label
        self.cache = cache

        if self.min_duration is None:
            self.generator_ = FixedDurationSequences(
                self.feature_extractor,
                duration=self.duration,
                step=(1 - self.overlap) * self.duration,
                batch_size=1 if self.cache else -1)
        else:
            self.generator_ = VariableDurationSequences(
                self.feature_extractor,
                max_duration=self.duration,
                min_duration=self.min_duration,
                batch_size=1 if self.cache else -1)

        # there is no need to cache preprocessed features
        # as the generator is iterated only once
        self.generator_.cache_preprocessed_ = False

        self.sequence_generator_ = self.iter_sequences(cache=self.cache)

        # consume first element of generator
        # this is meant to pre-generate all labeled sequences once and for all
        # and also to precompute the number of unique labels
        next(self.sequence_generator_)

    def _precompute(self, Xy_generator, cache):

        with h5py.File(cache, mode='w', libver='latest') as fp:

            # initialize with a fixed number of sequences
            n_sequences = 1000

            y = fp.create_dataset(
                'y', shape=(n_sequences, ),
                dtype=h5py.special_dtype(vlen=bytes),
                maxshape=(None, ))

            for i, (X_, y_) in enumerate(Xy_generator):

                if i == 0:
                    _, n_samples, n_features = X_.shape
                    X = fp.create_dataset(
                        'X', dtype=X_.dtype, compression='gzip',
                        shape=(n_sequences, n_samples, n_features),
                        chunks=(1, n_samples, n_features),
                        maxshape=(None, n_samples, n_features))

                # increase number of sequences on demand
                if i == n_sequences:
                    n_sequences = int(n_sequences * 1.1)
                    y.resize(n_sequences, axis=0)
                    X.resize(n_sequences, axis=0)

                # store current X, y in file
                y[i] = y_
                X[i] = X_

            # resize file to exactly match the number of sequences
            y.resize(i, axis=0)
            X.resize(i, axis=0)

    def iter_sequences(self, cache=None):

        # pre-generate all labeled sequences (from the whole training set)

        # in memory
        if cache is None:
            Xy_generator = self.generator_(self.file_generator)
            X, y = zip(*Xy_generator)
            X = np.vstack(X)
            y = np.hstack(y)

        # in HDF5 file
        elif not os.path.isfile(cache):
            Xy_generator = self.generator_(self.file_generator)
            self._precompute(Xy_generator, cache)

        if cache:
            fp = h5py.File(cache, mode='r', libver='latest')
            X = fp['X']
            y = fp['y']

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

        if cache:
            fp.close()

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


class DerivativeBatchGenerator(BaseBatchGenerator):
    """

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
        Defaults to 1.
    cache: str, optional
        Defaults to 'in-memory'
    """

    def __init__(self, feature_extractor, file_generator, compute_derivatives,
                 distance='angular', duration=3.0, min_duration=None,
                 per_label=3, per_fold=20, per_batch=12, n_threads=1,
                 cache=None):

        self.cache = cache

        self.sequence_generator_ = SequenceGenerator(
            feature_extractor, file_generator,
             duration=duration, min_duration=min_duration,
             per_label=per_label, cache=self.cache)

        self.n_labels = self.sequence_generator_.n_labels
        self.per_label = per_label
        self.per_fold = per_fold
        self.per_batch = per_batch
        self.n_threads = n_threads

        batch_size = self.per_label * self.per_fold * self.per_batch
        super(DerivativeBatchGenerator, self).__init__(
            self.sequence_generator_, batch_size=batch_size)

        self.compute_derivatives = compute_derivatives

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
