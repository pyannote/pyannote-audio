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

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input

from ..glue import Glue


def unitary_angular_triplet_loss(anchor, positive, negative):
    epsilon = 1e-6

    dotProdPosAnc = np.clip(np.sum(positive*anchor), -1.0, 1.0)
    dotProdNegAnc = np.clip(np.sum(negative*anchor), -1.0, 1.0)

    localCost = (np.arccos(dotProdPosAnc)-np.arccos(dotProdNegAnc)-np.pi/60.0)
    coeffSlope = 1.0
    coeffSlopeNegative = 1.0
    if (localCost < 0.0):
        coeffSlope = coeffSlopeNegative
    coeffSlopeInternal = 10.0
    localCost *= coeffSlopeInternal
    localCost = 1.0/(1.0 + np.exp(-localCost))

    dotProdPosAnc = 1-dotProdPosAnc*dotProdPosAnc
    dotProdNegAnc = 1-dotProdNegAnc*dotProdNegAnc
    if (dotProdPosAnc < epsilon): dotProdPosAnc = epsilon
    if (dotProdNegAnc < epsilon): dotProdNegAnc = epsilon

    derivCoeff = localCost*(1.0-localCost)*coeffSlope*coeffSlopeInternal
    localCost = coeffSlope*localCost+(coeffSlopeNegative-coeffSlope)*0.5

    derivativeAnchor = (-positive/np.sqrt(dotProdPosAnc)+negative/np.sqrt(dotProdNegAnc))*derivCoeff
    derivativePositive = -anchor/np.sqrt(dotProdPosAnc)*derivCoeff
    derivativeNegative = (anchor/np.sqrt(dotProdNegAnc))*derivCoeff

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]

def unitary_cosine_triplet_loss(anchor, positive, negative):
    dotProdPosAnc = np.sum(positive*anchor)
    dotProdNegAnc = np.sum(negative*anchor)

    localCost = -dotProdPosAnc+dotProdNegAnc-1.0/30.0
    # coeffSlope = 1.0
    # coeffSlopeNegative = 1.0
    # if (localCost < 0.0):
    #     coeffSlope = coeffSlopeNegative
    # coeffSlopeInternal = 10.0*np.pi/2.0
    # localCost *= coeffSlopeInternal
    localCost = 1.0/(1.0 + np.exp(-localCost))

    # derivCoeff = localCost*(1.0-localCost)*coeffSlope*coeffSlopeInternal

    derivCoeff = 1.0
    derivativeAnchor = (-positive+negative)*derivCoeff
    derivativePositive = -anchor*derivCoeff
    derivativeNegative = anchor*derivCoeff

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]

def unitary_euclidean_triplet_loss(anchor, positive, negative):
    localCost = np.sum(np.square(positive-anchor))-np.sum(np.square(negative-anchor))+0.2
    localCost = 1.0/(1.0 + np.exp(-localCost))

    derivativeAnchor = -2.0*(positive-negative)
    derivativePositive = -2.0*(anchor-positive)
    derivativeNegative = -2.0*(negative-anchor)

    return [localCost, derivativeAnchor, derivativePositive, derivativeNegative]

def batch_triplet_loss(inputs):
    embeddings = inputs[0]
    labels = inputs[1]
    distance = inputs[2]

    cost = 0.0
    derivative = 0.0*embeddings
    for ii in range(embeddings.shape[0]):
        for kk in range(embeddings.shape[0]):
            if ((kk != ii)and(labels[kk] == labels[ii])):
                for ll in range(embeddings.shape[0]):
                    if (labels[ll] != labels[kk]):
                        [local_cost, local_derivative_anchor, local_derivative_positive, local_derivative_negative] = distance(embeddings[ii,:], embeddings[kk,:], embeddings[ll,:])
                        cost += local_cost
                        derivative[ii,:] += local_derivative_anchor
                        derivative[kk,:] += local_derivative_positive
                        derivative[ll,:] += local_derivative_negative

    return [cost, derivative]


class BatchTripletLoss(Glue):
    """Batch version of the triplet loss for sequence embedding which
    is much faster

            anchor        |-----------|           |---------|
            input    -->  | embedding | --> a --> |         |
            sequence      |-----------|           |         |
                                                  |         |
            positive      |-----------|           | triplet |
            input    -->  | embedding | --> p --> |         | --> loss value
            sequence      |-----------|           |  loss   |
                                                  |         |
            negative      |-----------|           |         |
            input    -->  | embedding | --> n --> |         |
            sequence      |-----------|           |---------|

    Reference
    ---------
    Not yet written ;-)
    """
    def __init__(self, distance='angular'):
        super(BatchTripletLoss, self).__init__()
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
        return design_embedding(input_shape)

    def loss(self, y_true, y_pred):
        return self._derivative_loss(y_true, y_pred)

    def compute_cost_and_derivatives(self, embeddings, batch_labels, batch_size, nb_of_threads, pool):
        embeddings = embeddings.astype('float64')
        lines = []
        for jj in range(nb_of_threads):
            lines.append([embeddings[jj*batch_size:((jj+1)*batch_size),:],
                        batch_labels[jj*batch_size:((jj+1)*batch_size)],
                        self.loss_])

        costs = []
        derivatives = []
        for output in pool.imap(batch_triplet_loss, lines):
        # for line in lines:
        #     output = batch_triplet_loss(line)
            costs.append(output[0])
            derivatives.append(output[1])

        return [np.hstack(costs), np.vstack(derivatives)]
