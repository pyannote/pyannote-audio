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


class TripletLoss(Glue):
    """Triplet loss for sequence embedding

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

    Parameters
    ----------
    margin : float, optional
        Triplet loss margin. Defaults to 0.2.
    positive_only : boolean, optional
        When False, loss is d(a, p) - d(a, n) + margin.
        Default (True) is max(0, d(a, p) - d(a, n) + margin).
    distance: {'sqeuclidean', 'cosine'}
        Distance for which the embedding is optimized. Defaults to 'sqeuclidean'.

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017. https://arxiv.org/abs/1609.04301
    """
    def __init__(self, margin=0.2, positive_only=True, distance='sqeuclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.positive_only = positive_only
        self.distance = distance

    def _loss_sqeuclidean(self, inputs):
        p = K.sum(K.square(inputs[0] - inputs[1]), axis=-1, keepdims=True)
        n = K.sum(K.square(inputs[0] - inputs[2]), axis=-1, keepdims=True)
        loss = p + self.margin - n
        if self.positive_only:
            loss = K.maximum(0, loss)
        return loss

    def _loss_cosine(self, inputs):
        p = -K.sum(inputs[0] * inputs[1], axis=-1, keepdims=True)
        n = -K.sum(inputs[0] * inputs[2], axis=-1, keepdims=True)
        loss = p + self.margin - n
        if self.positive_only:
            loss = K.maximum(0, loss)
        return loss

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    @staticmethod
    def _identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def build_model(self, input_shape, design_embedding):
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

        anchor = Input(shape=input_shape, name="anchor")
        positive = Input(shape=input_shape, name="positive")
        negative = Input(shape=input_shape, name="negative")

        embedding = design_embedding(input_shape)
        embedded_anchor = embedding(anchor)
        embedded_positive = embedding(positive)
        embedded_negative = embedding(negative)

        mode = getattr(self, '_loss_' + self.distance)

        distance = merge(
            [embedded_anchor, embedded_positive, embedded_negative],
            mode=mode, output_shape=self._output_shape)

        model = Model(input=[anchor, positive, negative], output=distance)
        return model

    def loss(self, y_true, y_pred):
        return self._identity_loss(y_true, y_pred)

    def extract_embedding(self, from_model):
        return from_model.layers_by_depth[1][0]
