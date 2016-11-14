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
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import merge
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed


class TristouNet(object):
    """TristouNet sequence embedding

    LSTM ( » ... » LSTM ) » pooling › ( MLP › ... › ) MLP › normalize

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017.
    https://arxiv.org/abs/1609.04301

    Parameters
    ----------
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: {False, 'ave', 'concat'}, optional
        Defines how the output of forward and backward LSTMs are merged.
        'ave' stands for 'average', 'concat' (default) for concatenation.
        See keras.layers.wrappers.Bidirectional for more information.
        Use False to only use forward LSTMs.
    pooling: {'last', 'average'}
        Whether to use only the last output of the last LSTM ('last'),
        or to use its average output ('average', default).
    mlp: list, optional
        Number of units in additionnal stacked dense MLP layers.
        Defaults to [16, 16] (i.e. two dense MLP layers with 16 units)
    """

    def __init__(self, lstm=[16,], bidirectional='concat',
                 pooling='average', mlp=[16, 16]):

        super(TristouNet, self).__init__()
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.mlp = mlp

    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="input_sequence")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            if self.pooling == 'last':
                # only last LSTM should not return a sequence
                return_sequences = i+1 < n_lstm
            elif self.pooling == 'average':
                return_sequences = True
            else:
                raise NotImplementedError(
                    'unknown "{pooling}" pooling'.format(pooling=self.pooling))

            if i:
                # all but first LSTM
                lstm = LSTM(name='lstm_{i:d}'.format(i=i),
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)
            else:
                # first LSTM needs to be given the input shape
                lstm = LSTM(name='lstm_{i:d}'.format(i=i),
                               input_shape=input_shape,
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)

            if self.bidirectional:
                lstm = Bidirectional(lstm, merge_mode=self.bidirectional)

            x = lstm(x)

        if self.pooling == 'average':
            pooling = GlobalAveragePooling1D()
            x = pooling(x)

        # stack dense MLP layers
        for i, output_dim in enumerate(self.mlp):

            mlp = Dense(output_dim,
                        activation='tanh',
                        name='mlp_{i:d}'.format(i=i))
            x = mlp(x)

        # stack L2 normalization layer
        normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                           name="normalize")
        embeddings = normalize(x)

        return Model(input=inputs, output=embeddings)

    @property
    def output_dim(self):
        return self.mlp[-1]


class TrottiNet(object):
    """TrottiNet sequence embeddin

    LSTM ( » ... » LSTM ) » ( MLP » ... » ) MLP » pooling › normalize

    Parameters
    ----------
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: {False, 'ave', 'concat'}, optional
        Defines how the output of forward and backward LSTMs are merged.
        'ave' (default) stands for 'average', 'concat' for concatenation.
        See keras.layers.wrappers.Bidirectional for more information.
        Use False to only use forward LSTMs.
    mlp: list, optional
        Number of units of additionnal stacked dense MLP layers.
        Defaults to [16, 16] (i.e. add one dense MLP layer with 16 units)
    """

    def __init__(self, lstm=[16,], bidirectional='ave', mlp=[16, 16]):

        super(TrottiNet, self).__init__()
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.mlp = mlp

    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="input_sequence")
        x = inputs

        # stack (bidirectional) LSTM layers
        for i, output_dim in enumerate(self.lstm):

            if i:
                lstm = LSTM(name='lstm_{i:d}'.format(i=i),
                            output_dim=output_dim,
                            return_sequences=True,
                            activation='tanh',
                            dropout_W=0.0,
                            dropout_U=0.0)
            else:
                # we need to provide input_shape to first LSTM
                lstm = LSTM(name='lstm_{i:d}'.format(i=i),
                            input_shape=input_shape,
                            output_dim=output_dim,
                            return_sequences=True,
                            activation='tanh',
                            dropout_W=0.0,
                            dropout_U=0.0)

            if self.bidirectional:
                lstm = Bidirectional(lstm, merge_mode=self.bidirectional)

            x = lstm(x)

        # stack dense MLP layers
        for i, output_dim in enumerate(self.mlp):

            mlp = Dense(output_dim,
                        activation='tanh',
                        name='mlp_{i:d}'.format(i=i))

            x = TimeDistributed(mlp)(x)

        # average pooling
        pooling = GlobalAveragePooling1D(name='pooling')
        x = pooling(x)

        # L2 normalization layer
        normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                           name="normalize")
        embeddings = normalize(x)

        return Model(input=inputs, output=embeddings)

    @property
    def output_dim(self):
        return self.mlp[-1]

class MultiLevelTrottiNet(object):
    """MultiLevelTristouNet sequence embedding is a multi-level version of TristouNet

    Parameters
    ----------
    input_shape : (n_frames, n_features) tuple
        Shape of input sequence.
    optimizer: optimizer for
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs
    """

    def __init__(self, lstm=[16,8,8], dense=[], bidirectional=True):

        self.lstm = lstm
        self.dense = dense
        self.bidirectional = bidirectional

    def __call__(self, input_shape):
        inputs = Input(shape=input_shape, name="input_sequence")

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):
            if i:
                # all but first LSTM
                lstm_layer = LSTM(name='lstm_{i:d}'.format(i=i),
                                output_dim=output_dim,
                                return_sequences=True,
                                activation='tanh',
                                dropout_W=0.0,
                                dropout_U=0.0)

                if self.bidirectional:
                    lstm_out = Bidirectional(lstm_layer, merge_mode='ave')(lstm_out)
                else:
                    lstm_out = lstm_layer(lstm_out)
                multi_level_lstm = merge([multi_level_lstm, lstm_out], mode='concat', concat_axis=-1)
            else:
                lstm_layer = LSTM(name='lstm_{i:d}'.format(i=i),
                                input_shape=input_shape,
                                output_dim=output_dim,
                                return_sequences=True,
                                activation='tanh',
                                dropout_W=0.0,
                                dropout_U=0.0)
                # first forward LSTM needs to be given the input shape
                if self.bidirectional:
                    # first backward LSTM needs to be given the input shape
                    # AND to be told to process the sequence backward
                    lstm_out = Bidirectional(lstm_layer, merge_mode='ave')(inputs)
                else:
                    lstm_out = lstm_layer(inputs)
                multi_level_lstm = lstm_out

        if (len(self.dense) > 0):
            for i, output_dim in enumerate(self.dense):
                multi_level_lstm = TimeDistributed(Dense(output_dim,
                          activation='tanh',
                          name='dense_{i:d}'.format(i=i)))(multi_level_lstm)
     
        multi_level_lstm_avg = GlobalAveragePooling1D()(multi_level_lstm)

        # stack L2 normalization layer
        embeddings = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                            name="embedding_output")(multi_level_lstm_avg)

        return Model(input=[inputs], output=[embeddings])

    @property
    def output_dim(self):
        if len(self.dense):
            return self.dense[-1]
        return np.sum(self.lstm)
