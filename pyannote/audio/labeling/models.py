#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..train.utils import get_info
from torch.nn.utils.rnn import PackedSequence

from . import TASK_CLASSIFICATION
from . import TASK_MULTI_LABEL_CLASSIFICATION
from . import TASK_REGRESSION


class StackedRNN(nn.Module):
    """Stacked recurrent neural network

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_classes : int
        Set number of classes.
    task_type : {TASK_CLASSIFICATION, TASK_MULTI_LABEL_CLASSIFICATION,
            TASK_REGRESSION}
        Depending on which task is adressed, the final activation will vary.
        Classification relies on log-softmax, multi-label classificatition and
        regression use sigmoid.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

    def __init__(self, n_features, n_classes, task_type, instance_normalize=False,
                 rnn='LSTM', recurrent=[16,], bidirectional=False,
                 linear=[16, ]):

        super(StackedRNN, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.task_type = task_type
        if task_type not in {TASK_CLASSIFICATION,
                        TASK_MULTI_LABEL_CLASSIFICATION,
                        TASK_REGRESSION}:

            msg = (f"`task_type` must be one of {TASK_CLASSIFICATION}, "
                   f"{TASK_MULTI_LABEL_CLASSIFICATION} or {TASK_REGRESSION}.")
            raise ValueError(msg)

        self.instance_normalize = instance_normalize

        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional

        self.linear = linear

        self.num_directions_ = 2 if self.bidirectional else 1

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # create final classification layer
        self.final_layer_ = nn.Linear(input_dim, self.n_classes)

    def get_loss(self):
        """Return loss function (with support for class weights)

        Returns
        -------
        loss_func : callable
            Function f(input, target, weight=None) -> loss value
        """

        if self.task_type == TASK_CLASSIFICATION:
            return F.nll_loss

        if self.task_type == TASK_MULTI_LABEL_CLASSIFICATION:
            return F.binary_cross_entropy

        if self.task_type == TASK_REGRESSION:
            def mse_loss(input, target, weight=None):
                return F.mse_loss(input, target)
            return mse_loss

    def forward(self, sequences):

        if isinstance(sequences, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(sequences)

        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequences

        if self.instance_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            output = output.transpose(1, 2)

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (output[:, :, :hidden_dim] + \
                               output[:, :, hidden_dim:])

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # apply final classification layer
        output = self.final_layer_(output)

        if self.task_type == TASK_CLASSIFICATION:
            return torch.log_softmax(output, dim=2)

        elif self.task_type == TASK_MULTI_LABEL_CLASSIFICATION:
            return torch.sigmoid(output)

        elif self.task_type == TASK_REGRESSION:
            return torch.sigmoid(output)


class ConvRNN(nn.Module):
    """1D convolutional network followed by a stacked recurrent neural network

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_classes : int
        Set number of classes.
    task_type : {TASK_CLASSIFICATION, TASK_MULTI_LABEL_CLASSIFICATION,
            TASK_REGRESSION}
        Depending on which task is adressed, the final activation will vary.
        Classification relies on log-softmax, multi-label classificatition and
        regression use sigmoid.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

    def __init__(self, n_features, n_classes, task_type, instance_normalize=False,
                 convolutionnal=[16,], rnn='LSTM', recurrent=[16,], bidirectional=False,
                 linear=[16, ], kernel_size=32):

        super(ConvRNN, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.task_type = task_type
        if task_type not in {TASK_CLASSIFICATION,
                        TASK_MULTI_LABEL_CLASSIFICATION,
                        TASK_REGRESSION}:

            msg = (f"`task_type` must be one of {TASK_CLASSIFICATION}, "
                   f"{TASK_MULTI_LABEL_CLASSIFICATION} or {TASK_REGRESSION}.")
            raise ValueError(msg)

        self.instance_normalize = instance_normalize
        self.convolutionnal = convolutionnal
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.linear = linear
        self.kernel_size = kernel_size

        self.num_directions_ = 2 if self.bidirectional else 1

        input_dim = self.n_features
        # create 1D convolutional network
        # input_dim = self.n_features
        self.convolutionnal_layer_ = []
        self.convolutionnal_layer_ = nn.Conv2d(in_channels=1, out_channels=input_dim,
                                               kernel_size=[self.kernel_size, 1])
        self.batch_norm_ = nn.BatchNorm2d(self.n_features)
        self.relu_ = nn.ReLU()
        self.maxpool_ = nn.MaxPool2d(kernel_size=[1, input_dim-self.kernel_size+1])

        # create list of recurrent layers
        self.recurrent_layers_ = []
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # create final classification layer
        self.final_layer_ = nn.Linear(input_dim, self.n_classes)

    def get_loss(self):
        """Return loss function (with support for class weights)

        Returns
        -------
        loss_func : callable
            Function f(input, target, weight=None) -> loss value
        """

        if self.task_type == TASK_CLASSIFICATION:
            return F.nll_loss

        if self.task_type == TASK_MULTI_LABEL_CLASSIFICATION:
            return F.binary_cross_entropy

        if self.task_type == TASK_REGRESSION:
            def mse_loss(input, target, weight=None):
                return F.mse_loss(input, target)
            return mse_loss

    def forward(self, sequences):

        if isinstance(sequences, PackedSequence):
            msg = (f'{self.__class__.__name__} does not support batches '
                   f'containing sequences of variable length.')
            raise ValueError(msg)

        batch_size, n_features, device = get_info(sequences)

        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequences

        if self.instance_normalize:
            output = output.transpose(1, 2)
            output = F.instance_norm(output)
            output = output.transpose(1, 2)

        # get convolutional layer
        # [batch_size, seq_len, nb_mel] to [batch_size, 1, nb_mel, seq_len]
        output = output.transpose(1, 2).unsqueeze(1).contiguous()
        output = self.convolutionnal_layer_(output)
        output = self.batch_norm_(output)
        output = self.relu_(output)
        output = output.transpose(1, 2).transpose(1, 3).contiguous()
        output = self.maxpool_(output)
        output = output.squeeze(dim=3)

        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (output[:, :, :hidden_dim] + \
                               output[:, :, hidden_dim:])

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = torch.tanh(output)

        # apply final classification layer
        output = self.final_layer_(output)

        if self.task_type == TASK_CLASSIFICATION:
            return torch.log_softmax(output, dim=2)

        elif self.task_type == TASK_MULTI_LABEL_CLASSIFICATION:
            return torch.sigmoid(output)

        elif self.task_type == TASK_REGRESSION:
            return torch.sigmoid(output)
