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


from .triplet_loss import TripletLoss
from ..base import value_and_multigrad
from autograd import numpy as ag_np

import numpy as np
import h5py

from pyannote.generators.indices import random_label_index
from pyannote.generators.batch import batchify

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
import keras.models
from pyannote.audio.optimizers import SSMORMS3
from pyannote.audio.embedding.losses import precomputed_gradient_loss
from pyannote.audio.callback import LoggingCallback
from pyannote.audio.keras_utils import CUSTOM_OBJECTS
from pyannote.core.util import pairwise
from pyannote.audio.embedding.utils import cdist


class CenterLoss(TripletLoss):
    """

    loss = d(anchor, center) - d(anchor, other_center)

    * with 'positive' clamping:
        loss = max(0, loss + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * (loss - margin x D))

    where d(x, y) varies in range [0, D] (e.g. D=2 for euclidean distance).

    Parameters
    ----------
    metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'angular'.
    margin: float, optional
        Margin factor. Defaults to 0.
    clamp: {None, 'positive', 'sigmoid'}, optional
        If 'positive', loss = max(0, loss)
        If 'sigmoid' (default), loss = sigmoid(loss)
    per_label : int, optional
        Number of sequences per speaker. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to 20.
    per_batch : int, optional
        Number of folds per batch. Defaults to 1.
        Has no effect when `per_fold` is not provided.
    sampling : {'all', 'semi-hard', 'hard', 'hardest'}
        Negative sampling strategy. Defaults to 'all'.
    n_negative : int, optional
        Number of other centers to sample per (anchor, center) pair.
        Defaults to sample every valid centers.
    learn_to_aggregate : boolean, optional
    gradient_factor : float, optional
        Multiply gradient by this number. Defaults to 1.
    batch_size : int, optional
        Batch size. Defaults to 32.
    update_centers : {'batch', 'all'}
        Whether to only update centers in current 'batch' (default), or to
        update 'all' centers (even though they are not part of current batch).
    """

    WEIGHTS_H5 = LoggingCallback.WEIGHTS_H5[:-3] + '.centers.h5'
    CENTERS_TXT = '{log_dir}/centers.txt'

    def __init__(self, metric='angular', margin=0.0, clamp='sigmoid',
                 per_batch=1, per_fold=20, per_label=3,
                 update_centers='batch', learn_to_aggregate=False, **kwargs):

        super(CenterLoss, self).__init__(
            metric=metric, margin=margin, clamp=clamp,
            per_label=per_label, per_fold=per_fold, per_batch=per_batch,
            learn_to_aggregate=learn_to_aggregate, **kwargs)

        self.update_centers = update_centers

    def on_train_begin(self, logs=None):

        # number of classes
        n_classes = logs['n_classes']

        if logs['restart']:

            weights_h5 = self.WEIGHTS_H5.format(log_dir=logs['log_dir'],
                                                epoch=logs['epoch'])

            self.centers_ = keras.models.load_model(
                weights_h5, custom_objects=CUSTOM_OBJECTS,
                compile=True)

        else:

            # dimension of embedding space
            output_dim = self.model.output_shape[-1]

            # centers model
            trigger = Input(shape=(n_classes, ), name="trigger")
            x = Dense(output_dim, activation='linear', name='dense')(trigger)
            centers = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                             output_shape=(output_dim, ),
                             name="centers")(x)

            self.centers_ = Model(inputs=trigger, outputs=centers)
            self.centers_.compile(optimizer=SSMORMS3(),
                                  loss=precomputed_gradient_loss)

            # save list of classes
            centers_txt = self.CENTERS_TXT.format(**logs)
            with open(centers_txt, mode='w') as fp:
                for label in logs['classes']:
                    fp.write('{label}\n'.format(label=label.decode('utf8')))

        self.trigger_ = np.eye(n_classes)
        self.fC_ = self.centers_.predict(self.trigger_).astype(self.float_autograd_)


    def on_batch_end(self, batch_index, logs=None):
        self.centers_.train_on_batch(
            self.trigger_,
            logs['center_gradient'].astype(self.float_backend_))

        self.fC_ = self.centers_.predict(self.trigger_).astype(self.float_autograd_)

    def on_epoch_end(self, epoch, logs=None):
        """Save center weights after each epoch"""

        weights_h5 = self.WEIGHTS_H5.format(log_dir=logs['log_dir'],
                                            epoch=epoch)
        keras.models.save_model(self.centers_, weights_h5,
                                overwrite=logs['restart'],
                                include_optimizer=(epoch % 10 == 0))

        # TODO | plot distribution of distances between centers

    def loss_y_fold(self, fX, y, fC):
        """Differentiable loss

        Parameters
        ----------
        fX : (batch_size, n_dimensions) numpy array
            Embeddings.
        y : (batch_size, ) numpy array
            Labels.
        fC : (n_classes, n_dimensions) numpy array
            Centers.

        Returns
        -------
        loss : float
            Loss.
        n_comparisons : int
        """

        loss = 0.
        n_comparisons = 0

        # compute distances between embeddings and centers
        distance = self.metric_(fX, other_embedding=fC)

        # compare to every center...
        if self.update_centers == 'all':
            centers = list(range(fC.shape[0]))

        # or just to the ones in current batch
        elif self.update_centers == 'batch':
            centers = list(np.unique(y))

        # consider every embedding as anchor
        for anchor, y_anchor in enumerate(y):

            # anchor is the index of current embedding
            # y_anchor is the index of corresponding center

            for y_center in centers:

                if y_center == y_anchor:
                    continue

                # y_center is the index of another center

                loss_ = distance[anchor, y_anchor] - \
                        distance[anchor, y_center]

                if self.clamp == 'positive':
                    loss_ = loss_ + self.margin * self.metric_max_
                    loss_ = ag_np.maximum(loss_, 0.)

                elif self.clamp == 'sigmoid':
                    loss_ = loss_ - self.margin * self.metric_max_
                    loss_ = 1. / (1. + ag_np.exp(-10. * loss_))

                # do not use += because autograd does not support it
                loss = loss + loss_

                n_comparisons = n_comparisons + 1

        return loss, n_comparisons

    def loss_and_grad(self, batch, embedding):

        if self.learn_to_aggregate:
            fX = self.embed(embedding, batch['X'], internal=True)
            func = value_and_multigrad(self.loss_z, argnums=[0, 3])
            loss, (fX_grad, fC_grad) = func(fX, batch['y'],
                                            batch['n'], self.fC_)
            fX_grad = fX_grad[:, 0, :]

        else:
            fX = self.embed(embedding, batch['X'], internal=False)
            func = value_and_multigrad(self.loss_y, argnums=[0, 2])
            loss, (fX_grad, fC_grad) = func(fX, batch['y'], self.fC_)

        return {'loss': loss,
                'gradient': fX_grad,
                'center_gradient': fC_grad}


class NearestCenterLoss(CenterLoss):
    """Same as center loss except the generator is a bit smarter:
    it iterates through each center in (random) order, and selects samples
    from closest other centers
    """

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

        # iterates over sequences of class jC
        # in random order, and forever
        def class_generator(jC):
            indices = np.where(y == jC)[0]
            while True:
                np.random.shuffle(indices)
                for i in indices:
                    yield i

        def generator():

            centers = np.arange(n_classes)
            class_generators = [class_generator(jC) for jC in centers]

            previous_label = None

            while True:

                # loop over each centers in random order
                np.random.shuffle(centers)
                for iC in centers:

                    try:
                        # get "per_fold" closest centers to current centers
                        distances = cdist(self.fC_[iC, np.newaxis], self.fC_,
                                          metric=self.metric)[0]
                    except AttributeError as e:
                        # when on_train_begin hasn't been called yet,
                        # attribute fC_ doesn't exist --> fake it
                        distances = np.random.rand(len(centers))
                        distances[iC] = 0.

                    closest_centers = np.argpartition(
                        distances, self.per_fold)[:self.per_fold]

                    # corner case where last center of previous loop
                    # is the same as first center of current loop
                    if closest_centers[0] == previous_label:
                        closest_centers[:-1] = closest_centers[1:]
                        closest_centers[-1] = previous_label

                    for jC in closest_centers:
                        for _ in range(self.per_label):
                            i = next(class_generators[jC])
                            yield {'X': h5_X[i], 'y': y[i]}
                        previous_label = jC

        signature = {'X': {'type': 'ndarray'},
                     'y': {'type': 'ndarray'}}
        batch_size = self.per_batch * self.per_fold * self.per_label
        batch_generator = batchify(generator(), signature,
                                   batch_size=batch_size)

        # each fold contains one center and its `per_fold` closest centers
        # therefore, the only way to be sure that we've seen every class in
        # one epoch is to go through `n_classes` folds,
        # i.e. n_classes / per_batch batches
        batches_per_epoch = n_classes // self.per_batch

        return {'batch_generator': batch_generator,
                'batches_per_epoch': batches_per_epoch,
                'n_classes': n_classes,
                'classes': unique}
