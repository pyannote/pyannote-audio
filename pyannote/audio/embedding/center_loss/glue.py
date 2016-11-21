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
# Grégory GELLY
# Hervé BREDIN - http://herve.niderb.fr

import numpy as np
from functools import partial

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda

from pyannote.audio.embedding.glue import BatchGlue
from pyannote.audio.optimizers import SSMORMS3


def center_loss(inputs, centers=None, distance=None):
    """Compute embeddings and center derivatives

    Parameters
    ----------
    embeddings : (n_samples, n_dimensions) numpy array
    labels : (n_samples, ) numpy array
    center_labels : (n_centers, n_dimensions) numpy array
        n_centers <= n_labels
    centers : (n_labels, n_dimensions)
    distance : callable

    Returns
    -------
    cost : float
    d_embeddings : ()
        Embedding derivatives
    d_centers :
        Center derivatives
    """

    embeddings = inputs[0]
    labels = inputs[1]
    center_labels = inputs[2]

    cost = 0.0

    # embedding derivatives
    d_embeddings = 0.0 * embeddings

    # center derivatives
    d_centers = 0.0 * centers

    # loop on every embedding
    for ii, (embedding, label) in enumerate(zip(embeddings, labels)):

        # loop on every center
        for kk, true_center in enumerate(center_labels):

            # if current embedding belongs to current cluster
            if (true_center == label):

                # for every other center
                for ll, other_center in enumerate(center_labels):
                    if (other_center == true_center):
                        continue

                    [cost_, d_anchor_, d_positive_, d_negative_] = distance(
                        embedding, centers[true_center, :], centers[other_center, :])
                    cost += cost_

                    d_embeddings[ii, :] += d_anchor_
                    d_centers[true_center, :] += d_positive_
                    d_centers[other_center, :] += d_negative_

    return [cost, d_embeddings, d_centers]


class CenterLoss(BatchGlue):
    """Center loss for sequence embedding

    Parameters
    ----------
    per_label : int, optional
        Number of sequences per label. Defaults to 3.
    per_fold : int, optional
        Number of labels per fold. Defaults to 20.
    per_batch: int, optional
        Number of folds per batch. Defaults to 12.

    Reference
    ---------
    Not yet written ;-)
    """

    def _initialize_centers(self, n_labels, output_dim):
        trigger = Input(shape=(n_labels, ), name="trigger")
        x = Dense(output_dim, activation='linear', name='dense')(trigger)
        centers = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                         name="centers")(x)

        model = Model(input=trigger, output=centers)
        model.compile(optimizer=SSMORMS3(), loss=self.loss)
        return model

    def build_model(self, input_shape, design_embedding, n_labels=None, **kwargs):
        """Design the model for which the loss is optimized

        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.
        design_embedding : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the embedding as
            output.

        Returns
        -------
        model : Keras model

        See also
        --------
        An example of `design_embedding` is
        pyannote.audio.embedding.models.TristouNet.__call__
        """

        self.n_labels_ = n_labels
        self.centers_ = self._initialize_centers(
            self.n_labels_, design_embedding.output_dim)
        self.trigger_ = np.eye(self.n_labels_)
        return design_embedding(input_shape)

    def compute_derivatives(self, embeddings, labels):
        """Compute embedding derivatives

        Parameters
        ----------
        embeddings : (n_samples, output_dim)
        labels : (n_samples, )
            labels is expected to be (n_threads x batch_size)
        """
        embeddings = embeddings.astype('float64')
        current_centers = self.centers_.predict(self.trigger_).astype('float64')

        folds = []
        fold_size = self.per_fold * self.per_label

        for t in range(self.per_batch):
            fold_labels = labels[t * fold_size: (t+1) * fold_size]
            center_labels = np.unique(fold_labels)
            fX = embeddings[t * fold_size:(t+1) * fold_size]
            y = labels[t * fold_size:(t+1) * fold_size]
            folds.append([fX, y, center_labels])

        # self.loss_ and current_centers are shared
        # by all subsequent calls to 'center_loss'
        process_fold = partial(center_loss,
                               distance=self.loss_,
                               centers=current_centers)

        # TODO - use zip instead of this for loop
        costs = []
        derivatives = []
        centers_derivatives = 0.0 * current_centers
        for output in self.pool_.imap(process_fold, folds):
            costs.append(output[0])
            derivatives.append(output[1])
            centers_derivatives += output[2]

        # update centers
        self.centers_.train_on_batch(self.trigger_, centers_derivatives)

        return [np.hstack(costs), np.vstack(derivatives)]
