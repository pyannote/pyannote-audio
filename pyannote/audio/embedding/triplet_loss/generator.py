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

import warnings
import itertools
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.audio.generators.labels import FixedDurationSequences
from keras.callbacks import Callback
from keras.models import model_from_yaml
from pyannote.audio.embedding.base import SequenceEmbedding


class UpdateEmbedding(Callback):

    def __init__(self, generator, extract_embedding):
        super(UpdateEmbedding, self).__init__()
        self.generator = generator
        self.extract_embedding = extract_embedding

    def on_train_begin(self, logs={}):
        self.generator.update(self.model, self.extract_embedding)

    def on_batch_begin(self, batch, logs={}):
        self.generator.update(self.model, self.extract_embedding)


class TripletGenerator(object):
    """Triplet generator for triplet loss sequence embedding

    Generates ([Xa, Xp, Xn], 1) tuples where
      * Xa is the anchor sequence (e.g. by speaker S)
      * Xp is the positive sequence (also uttered by speaker S)
      * Xn is the negative sequence (uttered by a different speaker)

    and such that d(f(Xa), f(Xn)) < d(f(Xa), f(Xp)) + margin where
      * f is the current state of the embedding network (being optimized)
      * d is the (euclidean or cosine) distance
      * margin is the triplet loss margin (e.g. 0.2, typically)

    Parameters
    ----------
    extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC instance)
    file_generator: iterable
        File generator (the training set, typically)
    distance: {'sqeuclidean', 'cosine'}
        Distance for which the embedding is optimized. Defaults to 'sqeuclidean'.
    margin : float, optional
        Defaults to 0.2.
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    overlap: float, optional
        Sequence overlap ratio. Defaults to 0 (no overlap).
    per_label: int, optional
        Number of samples per label. Defaults to 40.
    per_fold: int, optional
        When provided, randomly split the training set into
        fold of `per_fold` labels (e.g. 40) after each epoch.
        Defaults to using the whole training set.
    batch_size: int, optional
        Batch size. Defaults to 32.
    """

    def __init__(self, extractor, file_generator,
                 distance='sqeuclidean', margin=0.2,
                 duration=3.0, overlap=0.8,
                 per_fold=0, per_label=40, batch_size=32):

        super(TripletGenerator, self).__init__()

        self.extractor = extractor
        self.file_generator = file_generator
        self.distance = distance
        self.margin = margin
        self.duration = duration
        self.overlap = overlap
        self.per_fold = per_fold
        self.per_label = per_label
        self.batch_size = batch_size

        self.generator_ = FixedDurationSequences(
            self.extractor,
            duration=self.duration,
            step=(1 - self.overlap) * self.duration,
            batch_size=-1)

        self.triplet_generator_ = self.iter_triplets()

        # consume first element of generator
        # this is meant to pre-generate all labeled sequences once and for all
        # and get the number of unique labels into self.n_labels
        next(self.triplet_generator_)

    def iter_triplets(self):

        # pre-generate all labeled sequences (from the whole training set)
        # this might be huge in memory
        X, y = [], []
        for batch_sequences, batch_labels in self.generator_(self.file_generator):
            X.append(batch_sequences)
            y.append(batch_labels)
        X = np.vstack(X)
        y = np.hstack(y)

        # unique labels
        unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
        self.n_labels = len(unique)

        # warn that some labels have very few training samples
        too_few_samples = np.sum(counts < self.per_label)
        if too_few_samples > 0:
            msg = '{n} labels (out of {N}) have less than {per_label} training samples.'
            warnings.warn(msg.format(n=too_few_samples,
                                     N=self.n_labels,
                                     per_label=self.per_label))

        # HACK (see __init__ for details on why this is done)
        yield

        # infinite loop
        while True:

            # shuffle labels
            shuffled_labels = np.random.choice(self.n_labels,
                                               size=self.n_labels,
                                               replace=False)

            if self.per_fold < 1:
                self.per_fold = self.n_labels

            # take them per_fold per per_fold
            for k in range(self.n_labels / self.per_fold):
                from_label = k * self.per_fold
                to_label = (k+1) * self.per_fold
                labels = shuffled_labels[from_label: to_label]

                # select min(per_label, count) sequences
                # at random for each label

                # per_label[i] contains the actual number of examples
                # available for ith label -- as it may actually be smaller
                # than the requested self.per_label for small classes (e.g.
                # for speakers without only a few seconds of speech)
                per_label = []

                # indices contains the list of indices of all sequences
                # to be used for later triplet selection
                indices = []

                for label in labels:

                    # number of available sequences for current label
                    # per_label.append(min(self.per_label, counts[label]))
                    per_label.append(self.per_label)

                    # NOTE the impact of choosing per_label instead of
                    # min(per_label, counts[label]) should be evaluated.
                    # indeed, for labels with a counts[label] smaller than
                    # per_label, the following 'choice' will repeat sequences.
                    # is this really an issue?

                    # randomly choose this many sequences
                    # from the set of available sequences
                    i = np.random.choice(
                        np.where(y == label)[0],
                        size=per_label[-1],
                        replace=True)

                    # append indices of selected sequences
                    indices.append(i)

                # after this line, per_label[i] will contain the position of
                # the first sequence of ith label so that the range
                # per_label[i]: per_label[i+1] points to the indices
                # corresponding to all sequences from ith label
                per_label = np.hstack([[0], np.cumsum(per_label)])

                # turn indices into a 1-dimensional numpy array.
                # combined with (above) per_label, it can be used
                # to get all indices of sequences from a given label
                indices = np.hstack(indices)

                # pre-compute pairwise distances d(f(X), f(X')) between every
                # pair (X, X') of selected sequences, where f is the current
                # state of the embedding being optimized, and d is the
                # euclidean distance

                # selected sequences
                sequences = X[indices]
                # their embeddings (using current state of embedding network)
                embeddings = self.embedding.transform(
                    sequences, batch_size=self.batch_size)

                # pairwise squared euclidean distances
                distances = squareform(pdist(embeddings, metric=self.distance))

                for i in range(self.per_fold):

                    positives = list(range(per_label[i], per_label[i+1]))
                    negatives = list(range(per_label[i])) + list(range(per_label[i+1], per_label[-1]))

                    # loop over all (anchor, positive) pairs for current label
                    for anchor, positive in itertools.combinations(positives, 2):

                        # find all negatives within the margin
                        d = distances[anchor, positive]
                        within_margin = np.where(
                            distances[anchor, negatives] < d + self.margin)[0]

                        # choose one at random (if at least one exists)
                        if len(within_margin) < 1:
                            continue
                        # TODO / add an option to choose the most difficult one
                        negative = negatives[np.random.choice(within_margin)]

                        yield [sequences[anchor], sequences[positive], sequences[negative]], 1

                        # FIXME -- exit this loop when an epoch has ended
                    # FIXME -- exit this loop when an epoch has ended
                # FIXME -- exit this loop when an epoch has ended

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.triplet_generator_)

    def get_shape(self):
        return self.generator_.get_shape()

    def signature(self):
        shape = self.get_shape()
        return (
            [
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape}
            ],
            {'type': 'boolean'}
        )

    def update(self, new_model, extract_embedding):

        # make a copy of current embedding
        embedding = extract_embedding(new_model)
        embedding_copy = model_from_yaml(embedding.to_yaml())
        embedding_copy.set_weights(embedding.get_weights())

        # update the embedding used by the generator
        sequence_embedding = SequenceEmbedding()
        sequence_embedding.embedding_ = embedding_copy
        self.embedding = sequence_embedding

    def callbacks(self, extract_embedding=None):
        callback = UpdateEmbedding(self, extract_embedding=extract_embedding)
        return [callback]


class TripletBatchGenerator(BaseBatchGenerator):
    """Triplet generator for triplet loss sequence embedding

    Generates ([Xa, Xp, Xn], 1) batch tuples where
      * Xa are anchor sequences (e.g. by speaker S)
      * Xp are positive sequences (also uttered by speaker S)
      * Xn are negative sequences (uttered by a different speaker)

    and such that d(f(Xa), f(Xn)) < d(f(Xa), f(Xp)) + margin where
      * f is the current state of the embedding network (being optimized)
      * d is the euclidean distance
      * margin is the triplet loss margin (e.g. 0.2, typically)

    Parameters
    ----------
    feature_extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC instance)
    file_generator: iterable
        File generator (the training set, typically)
    distance: {'sqeuclidean', 'cosine'}
        Distance for which the embedding is optimized. Defaults to 'sqeuclidean'.
    margin : float, optional
        Defaults to 0.2.
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    overlap: float, optional
        Sequence overlap ratio. Defaults to 0 (no overlap).
    per_label: int, optional
        Number of samples per label. Defaults to 40.
    per_fold: int, optional
        Randomly split the training set into disjoint folds of `per_fold`
        labels. Defaults to using one big fold.
    batch_size: int, optional
        Batch size. Defaults to 32.
    """
    def __init__(self, feature_extractor, file_generator,
                 distance='sqeuclidean', margin=0.2,
                 duration=3.0, overlap=0.5,
                 per_fold=0, per_label=40, batch_size=32):

        self.triplet_generator_ = TripletGenerator(
            feature_extractor, file_generator,
            margin=margin, distance=distance,
            duration=duration, overlap=overlap, 
            per_fold=per_fold, per_label=per_label, batch_size=batch_size)

        super(TripletBatchGenerator, self).__init__(
            self.triplet_generator_, batch_size=batch_size)

    def signature(self):
        return self.triplet_generator_.signature()

    def get_shape(self):
        return self.triplet_generator_.get_shape()

    @property
    def n_labels(self):
        return self.triplet_generator_.n_labels

    def callbacks(self, extract_embedding=None):
        return self.triplet_generator_.callbacks(
            extract_embedding=extract_embedding)
