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
import warnings


class TristouNet(nn.Module):
    """TristouNet sequence embedding

    RNN ( » ... » RNN ) » temporal pooling › ( MLP › ... › ) MLP › normalize

    Parameters
    ----------
    n_features : int
        Input feature dimension
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, 16].

    Reference
    ---------
    Hervé Bredin. "TristouNet: Triplet Loss for Speaker Turn Embedding."
    ICASSP 2017 (https://arxiv.org/abs/1609.04301)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[16], bidirectional=False,
                 pooling='sum', linear=[16, 16]):

        super(TristouNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.linear = [] if linear is None else linear

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional)
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

    @property
    def batch_first(self):
        return False

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return self.recurrent[-1]

    def forward(self, sequence):

        # check input feature dimension
        n_samples, batch_size, n_features = sequence.size()
        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence
        # n_samples, batch_size, n_features
        device = sequence.device

        # recurrent layers
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

        # average temporal pooling
        if self.pooling == 'sum':
            output = output.sum(dim=0)
        elif self.pooling == 'max':
            output, _ = output.max(dim=0)

        # batch_size, dimension

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = F.tanh(output)

        # batch_size, dimension

        # unit-normalize
        norm = torch.norm(output, 2, 1, keepdim=True)
        output = output / norm

        return output


class VGGVox(nn.Module):

    def __init__(self, n_features, output_dim=128):

        super(VGGVox, self).__init__()
        self.n_features = n_features
        self.output_dim = output_dim

        self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1)
        self.bn1_ = nn.BatchNorm2d(96)
        self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1)
        self.bn2_ = nn.BatchNorm2d(256)
        self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn3_ = nn.BatchNorm2d(256)
        self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn4_ = nn.BatchNorm2d(256)
        self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        self.bn5_ = nn.BatchNorm2d(256)
        self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
        self.fc6_ = nn.Conv2d(256, 4096, (9, 1), stride=(1, 1))
        self.bn6_ = nn.BatchNorm2d(4096)
        self.fc7_ = nn.Conv2d(4096, 1024, (1, 1), stride=(1, 1))
        self.bn7_ = nn.BatchNorm2d(1024)
        self.fc8_ = nn.Conv2d(1024, self.output_dim, (1, 1), stride=(1, 1))
        self.bn8_ = nn.BatchNorm2d(self.output_dim)

    def forward(self, sequences):
        """Embed sequences

        Parameters
        ----------
        sequences : torch.autograd.Variable (batch_size, n_samples, n_features)
            Batch of sequences.

        Returns
        -------
        embeddings : torch.autograd.Variable (batch_size, output_dim)
            Batch of embeddings.

        """

        batch_size, n_samples, n_features = sequences.size()
        x = torch.transpose(sequences, 1, 2).view(
            batch_size, 1, n_features, n_samples)
        x = F.relu(self.bn1_(self.conv1_(x)))
        x = self.mpool1_(x)
        x = F.relu(self.bn2_(self.conv2_(x)))
        x = self.mpool2_(x)
        x = F.relu(self.bn3_(self.conv3_(x)))
        x = F.relu(self.bn4_(self.conv4_(x)))
        x = F.relu(self.bn5_(self.conv5_(x)))
        x = self.mpool5_(x)
        x = F.relu(self.bn6_(self.fc6_(x)))
        x = torch.mean(x, dim=-1, keepdim=True)
        x = F.relu(self.bn7_(self.fc7_(x)))
        x = F.relu(self.bn8_(self.fc8_(x))).view(-1, self.output_dim)

        return x


class ClopiNet(nn.Module):
    """ClopiNet sequence embedding

    RNN          ⎤
      » RNN      ⎥ » MLP » Weight » temporal pooling › normalize
           » RNN ⎦

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [64, 64, 64], i.e. three recurrent layers with hidden dimension of 64.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    batch_normalize : boolean, optional
        Set to False to not apply batch normalization before embedding
        normalization. Defaults to True.
    normalize : {False, 'sphere', 'ball', 'ring'}, optional
        Normalize embeddings.
    weighted : bool, optional
        Add dimension-wise trainable weights. Defaults to False.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to none.
    internal : bool, optional
        Return sequence of internal embeddings. Defaults to False.
    attention : list of int, optional
        List of hidden dimensions of attention linear layers (e.g. [16, ]).
        Defaults to no attention.
    return_attention : bool, optional

    Usage
    -----
    >>> model = ClopiNet(n_features)
    >>> embedding = model(sequence)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[64, 64, 64], bidirectional=False,
                 pooling='sum', batch_normalize=True, normalize=False,
                 weighted=False, linear=None, internal=False, attention=None,
                 return_attention=False):

        super(ClopiNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.batch_normalize = batch_normalize
        self.normalize = normalize
        self.weighted = weighted
        self.linear = [] if linear is None else linear
        self.internal = internal
        self.attention = [] if attention is None else attention
        self.return_attention = return_attention

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # the output of recurrent layers are concatenated so the input
        # dimension of subsequent linear layers is the sum of their output
        # dimension
        input_dim = sum(self.recurrent)

        if self.weighted:
            self.alphas_ = nn.Parameter(torch.ones(input_dim))

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # batch normalization ~= embeddings whitening.
        if self.batch_normalize:
            self.batch_norm_ = nn.BatchNorm1d(input_dim, eps=1e-5,
                                              momentum=0.1, affine=False)

        if self.normalize in {'ball', 'ring'}:
            self.norm_batch_norm_ = nn.BatchNorm1d(1, eps=1e-5, momentum=0.1,
                                                   affine=False)

        # create attention layers
        self.attention_layers_ = []
        if not self.attention:
            return

        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.attention):
            attention_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('attention_{0}'.format(i), attention_layer)
            self.attention_layers_.append(attention_layer)
            input_dim = hidden_dim
        if input_dim > 1:
            attention_layer = nn.Linear(input_dim, 1, bias=True)
            self.add_module('attention_{0}'.format(len(self.attention)),
                            attention_layer)
            self.attention_layers_.append(attention_layer)

    @property
    def batch_first(self):
        return False

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return sum(self.recurrent)

    def forward(self, sequence):

        # check input feature dimension
        n_samples, batch_size, n_features = sequence.size()
        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence
        # n_samples, batch_size, n_features
        device = sequence.device

        if self.weighted:
            self.alphas_ = self.alphas_.to(device)

        outputs = []
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

            outputs.append(output)

        # concatenate outputs
        output = torch.cat(outputs, dim=2)
        # n_samples, batch_size, dimension

        if self.weighted:
            output = output * self.alphas_

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = F.tanh(output)

        # n_samples, batch_size, dimension

        if self.internal:
            if self.normalize:
                msg = 'did not normalize internal embeddings.'
                warnings.warn(msg, UserWarning)
            return output

        if self.attention_layers_:
            attn = sequence
            for layer, hidden_dim in zip(self.attention_layers_,
                                         self.attention + [1]):
                attn = layer(attn)
                attn = F.tanh(attn)
            attn = F.softmax(attn, dim=0)
            output = output * attn

        # average temporal pooling
        if self.pooling == 'sum':
            output = output.sum(dim=0)
        elif self.pooling == 'max':
            output, _ = output.max(dim=0)

        # batch_size, dimension

        # batch normalization
        if self.batch_normalize:
            output = self.batch_norm_(output)

        if self.normalize:
            norm = torch.norm(output, 2, 1, keepdim=True)

        if self.normalize == 'sphere':
            output = output / norm

        elif self.normalize == 'ball':
            output = output / norm * F.sigmoid(self.norm_batch_norm_(norm))

        elif self.normalize == 'ring':
            norm_ = self.norm_batch_norm_(norm)
            output = output / norm * (1 + F.sigmoid(self.norm_batch_norm_(norm)))

        # batch_size, dimension

        if self.return_attention:
            return output, attn

        return output
