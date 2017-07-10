#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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


from ..base import SequenceEmbedding
from autograd import numpy as ag_np
from autograd import value_and_grad

import itertools
import numpy as np
import pandas as pd
import h5py

from pyannote.generators.indices import random_label_index
from pyannote.generators.batch import batchify

from pyannote.core.util import pairwise
from random import shuffle


class TripletLoss(SequenceEmbedding):
    """

    loss = d(anchor, positive) - d(anchor, negative)

    * 'positive' clamping >= 0: loss = max(0, loss + margin)
    * 'sigmoid' clamping [0, 1]: loss = sigmoid(10 * (loss - margin))

    Parameters
    ----------
    metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'sqeuclidean'.
    margin: float, optional
        Defaults to 0.1.
    clamp: {None, 'positive', 'sigmoid'}, optional
        If 'positive' (default), loss = max(0, loss)
        If 'sigmoid', loss = sigmoid(loss)
    per_label : int, optional
        Number of sequences per speaker. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    per_batch : int, optional
        Number of folds per batch. Defaults to 1.
        Has no effect when `per_fold` is not provided.
    sampling : {'all', 'hard'}
        Negative sampling strategy.
    learn_to_aggregate : boolean, optional
    gradient_factor : float, optional
        Multiply gradient by this number. Defaults to 1.
    batch_size : int, optional
        Batch size. Defaults to 32.

    """

    def __init__(self, metric='sqeuclidean', margin=0.1, clamp='positive',
                 per_batch=1, per_label=3, per_fold=None, sampling='all',
                 learn_to_aggregate=False, **kwargs):

        self.margin = margin
        self.clamp = clamp
        self.per_batch = per_batch
        self.per_label = per_label
        self.per_fold = per_fold
        self.sampling = sampling
        self.learn_to_aggregate = learn_to_aggregate
        super(TripletLoss, self).__init__(**kwargs)

    def get_batch_generator(self, data_h5):
        if self.learn_to_aggregate:
            return self._get_batch_generator_z(data_h5)
        else:
            return self._get_batch_generator_y(data_h5)

    def _get_batch_generator_y(self, data_h5):
        """Get batch generator

        Parameters
        ----------
        data_h5 : str
            Path to HDF5 file containing precomputed sequences.
            It must have to aligned datasets 'X' and 'y'.

        Returns
        -------
        batch_generator : iterable
        batches_per_epoch : int
        n_classes : int
        """

        fp = h5py.File(data_h5, mode='r')
        h5_X = fp['X']
        h5_y = fp['y']

        # keep track of number of labels and rename labels to integers
        unique, y = np.unique(h5_y, return_inverse=True)
        n_classes = len(unique)

        index_generator = random_label_index(
            y, per_label=self.per_label, return_label=False)

        def generator():
            while True:
                i = next(index_generator)
                yield {'X': h5_X[i], 'y': y[i]}

        signature = {'X': {'type': 'ndarray'},
                     'y': {'type': 'ndarray'}}

        if self.per_fold is None:
            batch_size = n_classes * self.per_label
            batches_per_epoch = 1
        else:
            batch_size = self.per_batch * self.per_fold * self.per_label
            batches_per_epoch = n_classes // (self.per_batch * self.per_fold) + 1

        batch_generator = batchify(generator(), signature,
                                   batch_size=batch_size)

        return {'batch_generator': batch_generator,
                'batches_per_epoch': batches_per_epoch,
                'n_classes': n_classes,
                'classes': unique}

    def _get_batch_generator_z(self, data_h5):
        """"""

        fp = h5py.File(data_h5, mode='r')
        h5_X = fp['X']
        h5_y = fp['y']
        h5_z = fp['z']

        df = pd.DataFrame({'y': h5_y, 'z': h5_z})
        z_groups = df.groupby('z')

        y_groups = [group.y.iloc[0] for _, group in z_groups]

        # keep track of number of labels and rename labels to integers
        unique, y = np.unique(y_groups, return_inverse=True)
        n_classes = len(unique)

        index_generator = random_label_index(
            y, per_label=self.per_label,
            return_label=True, repeat=False)

        def generator():
            while True:
                # get next group
                i, label = next(index_generator)

                # select at most 10 sequences of current group
                selector = list(z_groups.get_group(i).index)
                selector = np.random.choice(selector,
                                            size=min(10, len(selector)),
                                            replace=False)

                X = np.array(h5_X[sorted(selector)])
                n = X.shape[0]
                yield {'X': X,
                       'y': label,
                       'n': n}

        signature = {'X': {'type': 'ndarray'},
                     'y': {'type': 'scalar'},
                     'n': {'type': 'complex'}}

        if self.per_fold is None:
            batch_size = n_classes * self.per_label
            batches_per_epoch = 1
        else:
            batch_size = self.per_batch * self.per_fold * self.per_label
            batches_per_epoch = n_classes // (self.per_batch * self.per_fold) + 1

        batch_generator = batchify(generator(), signature,
                                   batch_size=batch_size)

        return {'batch_generator': batch_generator,
                'batches_per_epoch': batches_per_epoch,
                'n_classes': n_classes,
                'classes': unique}

    def loss_and_grad(self, batch, embedding):

        if self.learn_to_aggregate:
            fX = self.embed(embedding, batch['X'], internal=True)
            loss, fX_grad = value_and_grad(
                self.loss_z, argnum=0)(fX, batch['y'], batch['n'])
            fX_grad = fX_grad[:, 0, :]

        else:
            fX = self.embed(embedding, batch['X'], internal=False)
            loss, fX_grad = value_and_grad(
                self.loss_y, argnum=0)(fX, batch['y'])

        return {'loss': loss, 'gradient': fX_grad}

    def loss_y(self, fX, y, *args):

        if self.per_fold is None:
            loss, n_comparisons = self.loss_y_fold(fX, y, *args)
            return loss / n_comparisons

        loss = 0.
        n_comparisons = 0
        groups = itertools.groupby(enumerate(y), lambda iv: iv[1])

        for i_fold in range(self.per_batch):

            # gather start/end (min, MAX) index of current fold
            m, M = len(y), 0
            for i_label in range(self.per_fold):
                current_label, indices = next(groups)
                indices = list(zip(*indices))[0]
                m = min(m, min(indices))
                M = max(M, max(indices))

            loss_fold, n_comparisons_fold = \
                self.loss_y_fold(fX[m: M+1], y[m: M+1], *args)
            loss = loss + loss_fold
            n_comparisons += n_comparisons_fold

        return loss / n_comparisons

    def triplet_loss(self, distance, anchor, positive, negative=None):

        loss = distance[anchor, positive]

        if negative is None:
            loss = loss - distance[anchor, :]
        else:
            loss = loss - distance[anchor, negative]

        if self.clamp == 'positive':
            loss = loss + self.margin * self.metric_max_
            loss = ag_np.maximum(loss, 0.)

        elif self.clamp == 'sigmoid':
            loss = loss - self.margin * self.metric_max_
            loss = 1. / (1. + ag_np.exp(-10. * loss))

        return loss

    def triplet_sampling(self, y, anchor, positive, distance=None):

        if self.sampling == 'all':
            return self.triplet_sampling_all(y, anchor, positive)

        elif self.sampling == 'hard':
            return self.triplet_sampling_hard(y, anchor, positive,
                                              distance=distance)

        elif self.sampling == 'hardest':
            raise NotImplementedError('')

        elif self.sampling == 'semi-hard':
            raise NotImplementedError('')

    def triplet_sampling_all(self, y, anchor, positive, **kwargs):
        for negative, y_negative in enumerate(y):
            if y_negative == y[anchor]:
                continue
            yield negative

    def triplet_sampling_hard(self, y, anchor, positive, distance=None):
        """Choose negative at random such that
        d(anchor, negative) < d(anchor, positive) + margin
        """

        # find hard cases (loss > 0)
        loss = self.triplet_loss(distance, anchor, positive)
        hard_cases = np.where(loss > 0)[0]

        # choose at random == shuffle and choose the first one
        shuffle(hard_cases)
        for negative in hard_cases:
            # make sure it is not actually a positive sample
            if y[negative] != y[anchor]:
                yield negative
                break

    def loss_y_fold(self, fX, y):
        """Differentiable loss

        Parameters
        ----------
        fX : (batch_size, n_dimensions) numpy array
            Embeddings.
        y : (batch_size, ) numpy array
            Labels.

        Returns
        -------
        loss : float
            Loss.
        """

        loss = 0.
        n_comparisons = 0

        distance = self.metric_(fX)

        # consider every embedding as anchor
        for anchor, y_anchor in enumerate(y):

            # consider every other embedding with the same label as positive
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative in self.triplet_sampling(y, anchor, positive,
                                                      distance=distance):

                    loss_ = self.triplet_loss(distance,
                                              anchor,
                                              positive,
                                              negative)

                    # do not use += because autograd does not support it
                    loss = loss + loss_

                    n_comparisons = n_comparisons + 1

        return loss, n_comparisons

    def loss_z(self, fX, y, n, *args):
        """Differentiable loss

        Parameters
        ----------
        fX : np.array (n_sequences, n_samples, n_dimensions)
            Stacked groups of internal embeddings.
        y : (batch_size, ) numpy array
            Label of each group.
        n :  (batch_size, ) numpy array
            Number of sequences per group (np.sum(n) == n_sequences)

        Returns
        -------
        loss : float
            Loss.
        """

        indices = np.hstack([[0], np.cumsum(n)])
        fX_sum = ag_np.stack([ag_np.sum(ag_np.sum(fX[i:j], axis=0), axis=0)
                              for i, j in pairwise(indices)])
        return self.loss_y(self.l2_normalize(fX_sum), y, *args)
