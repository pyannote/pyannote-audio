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


from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import merge
from keras.layers.wrappers import TimeDistributed


class StackedLSTM(object):
    """Stacked LSTM sequence labeling

    Parameters
    ----------
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [16, ] (i.e. one LSTM with output dimension 16)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs
    dense: list, optional
        Number of units of additionnal stacked dense layers.
        Defaults to [16, ] (i.e. add one dense layer with 16 units)
    n_classes : int, optional
        Number of output classes. Defaults to 2 (binary classification).
    """
    def __init__(self, lstm=[16,], bidirectional=True,
                 dense=[16,], n_classes=2):

        super(StackedLSTM, self).__init__()
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.dense = dense
        self.n_classes = n_classes

    def __call__(self, input_shape):
        """Design labeling

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="labeling_input")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            if i:
                # all but first LSTM
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               output_dim=output_dim,
                               return_sequences=True,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(forward)

                if self.bidirectional:
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    output_dim=output_dim,
                                    return_sequences=True,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(backward)
            else:
                # first LSTM
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               output_dim=output_dim,
                               return_sequences=True,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(x)

                if self.bidirectional:
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    go_backwards=True,
                                    output_dim=output_dim,
                                    return_sequences=True,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(x)

        # concatenate forward and backward
        if self.bidirectional:
            x = merge([forward, backward], mode='concat', concat_axis=2)
        else:
            x = forward

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            x = TimeDistributed(Dense(output_dim,
                                      activation='tanh',
                                      name='dense_{i:d}'.format(i=i)))(x)

        # one dimension per class
        outputs = TimeDistributed(Dense(self.n_classes,
                                        activation='softmax',
                                        name="labeling_output"))(x)

        return Model(input=inputs, output=outputs)
