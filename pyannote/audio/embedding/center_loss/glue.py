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

import keras.backend as K
from keras.models import Model

from keras.layers import Input
from keras.layers import merge

from ..glue import Glue

import numpy as np
from pyannote.audio.embedding.batch_triplet_loss.glue import unitary_angular_triplet_loss,unitary_cosine_triplet_loss,unitary_euclidean_triplet_loss
from pyannote.audio.embedding.center_loss.models import CentersEmbeddings
from pyannote.audio.optimizers import SSMORMS3

def center_loss(inputs):
    labels = inputs[1]
    embeddings = inputs[0]
    labels2 = inputs[3]
    embeddings2 = inputs[2]
    distance = inputs[4]

    cost = 0.0
    derivative = 0.0*embeddings
    derivative2 = 0.0*embeddings2
    for ii in range(embeddings.shape[0]):
        for kk in range(labels2.shape[0]):
            if (labels2[kk] == labels[ii]):
                for ll in range(labels2.shape[0]):
                    if (labels2[ll] != labels2[kk]):
                        [local_cost, local_derivative_anchor, local_derivative_positive, local_derivative_negative] = distance(embeddings[ii,:], embeddings2[labels2[kk],:], embeddings2[labels2[ll],:])
                        cost += local_cost
                        derivative[ii,:] += local_derivative_anchor
                        derivative2[labels2[kk],:] += local_derivative_positive
                        derivative2[labels2[ll],:] += local_derivative_negative

    return [cost, derivative, derivative2]


class CenterLoss(Glue):
    """Center loss for sequence embedding

            anchor        |-----------|           |---------|
            input    -->  | embedding | --> a --> |         |
            sequence      |-----------|           |         |
                                                  |         |
            target        |-----------|           | triplet |
            center   -->  | embedding | --> p --> |         | --> loss value
            index         |-----------|           |  loss   |
                                                  |         |
            negative      |-----------|           |         |
            centers  -->  | embedding | --> n --> |         |
            indices       |-----------|           |---------|

    Reference
    ---------
    Not yet written ;-)
    """
    def __init__(self, distance='angular'):
        super(CenterLoss, self).__init__()
        if (distance == 'angular'):
            self.loss_ = unitary_angular_triplet_loss
        elif (distance == 'cosine'):
            self.loss_ = unitary_cosine_triplet_loss
        elif (distance == 'euclidean'):
            self.loss_ = unitary_euclidean_triplet_loss
        else:
            raise NotImplementedError(
                'unknown "{distance}" distance'.format(distance=distance))

    @staticmethod
    def _derivative_loss(y_true, y_pred):
        return K.sum((y_pred * y_true), axis=-1)

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    def build_model(self, input_shape, design_embedding, n_labels):
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
        design_center = CentersEmbeddings(output_dim=design_embedding.output_dim)
        self.centers = design_center((n_labels,))
        self.centers.compile(optimizer=SSMORMS3(), loss=self.loss)
        self.centers.summary()
        self.center_trigger = np.eye(n_labels)

        return design_embedding(input_shape)

    def loss(self, y_true, y_pred):
        return self._derivative_loss(y_true, y_pred)

    def extract_embedding(self, from_model):
        return from_model

    def compute_cost_and_derivatives(self, embeddings, batch_labels, batch_size, nb_of_threads, pool):
        embeddings = embeddings.astype('float64')
        embeddingscenters = self.centers.predict(self.center_trigger).astype('float64')

        lines = []
        for jj in range(nb_of_threads):
            center_labels = np.unique(batch_labels[jj*batch_size:((jj+1)*batch_size)])
            lines.append([embeddings[jj*batch_size:((jj+1)*batch_size),:],
                    batch_labels[jj*batch_size:((jj+1)*batch_size)],
                    embeddingscenters, center_labels, self.loss_])

        costs = []
        derivatives = []
        centers_derivatives = 0.0*embeddingscenters
        for output in pool.imap(center_loss, lines):
        # for line in lines:
        #     output = center_loss(line)
            costs.append(output[0])
            derivatives.append(output[1])
            centers_derivatives += output[2]
        
        self.centers.train_on_batch(self.center_trigger, centers_derivatives)

        return [np.hstack(costs), np.vstack(derivatives)]


