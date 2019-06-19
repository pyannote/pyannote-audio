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
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from ...train.utils import get_info
from ...train.utils import map_packed
from ...train.utils import pool_packed
from ...train.utils import operator_packed


class ClopiNet(nn.Module):
    """ClopiNet sequence embedding

    RNN          ⎤
      » RNN      ⎥ » MLP » Weight » temporal pooling › normalize
           » RNN ⎦

    Parameters
    ----------
    specifications : `dict`
        Batch specifications:
            {'X': {'dimension': n_features}}
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [256, 256, 256], i.e. three recurrent layers with hidden dimension of 64.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    instance_normalize : boolean, optional
        Apply mean/variance normalization on input sequences.
    batch_normalize : boolean, optional
        Set to False to not apply batch normalization before embedding
        normalization. Defaults to True.
    normalize : {False, 'sphere', 'ball', 'ring'}, optional
        Normalize embeddings.
    weighted : bool, optional
        Add dimension-wise trainable weights. Defaults to False.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to none.
    attention : list of int, optional
        List of hidden dimensions of attention linear layers (e.g. [16, ]).
        Defaults to no attention.

    Usage
    -----
    >>> model = ClopiNet(n_features)
    >>> embedding = model(sequence)
    """

    supports_packed = True

    def __init__(self, specifications,
                 rnn='LSTM', recurrent=[256, 256, 256], bidirectional=False,
                 pooling='sum', instance_normalize=False, batch_normalize=True,
                 normalize=False, weighted=False, linear=None, attention=None):

        super(ClopiNet, self).__init__()

        self.specifications = specifications
        self.n_features_ = specifications['X']['dimension']
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.instance_normalize = instance_normalize
        self.batch_normalize = batch_normalize
        self.normalize = normalize
        self.weighted = weighted
        self.linear = [] if linear is None else linear
        self.attention = [] if attention is None else attention

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features_
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
            # TODO. use nn.ModuleList instead
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)

        # the output of recurrent layers are concatenated so the input
        # dimension of subsequent linear layers is the sum of their output
        # dimension
        input_dim = sum(self.recurrent) * (2 if self.bidirectional else 1)

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

        input_dim = self.n_features_
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
    def dimension(self):
        if self.linear:
            return self.linear[-1]
        return sum(self.recurrent) * (2 if self.bidirectional else 1)

    def forward(self, sequences):
        """Forward pass

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.Tensor`
                    or `PackedSequence`
            Batch of sequences. Variable length is supported through
            `PackedSequence` instance but fixed length will be much faster.

        Returns
        -------
        embeddings : (batch_size, dimension) `torch.Tensor`
            Embeddings.
        """

        batch_size, n_features, device = get_info(sequences)
        if n_features != self.n_features_:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features_))

        output = sequences

        if self.instance_normalize:
            func = lambda b: F.instance_norm(b.transpose(1, 2)).transpose(1, 2)
            output = map_packed(func, output)
            # same as F.instance_norm(output) it supports PackedSequence

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

            # apply current recurrent layer and get output sequences
            output, _ = layer(output, hidden)

            outputs.append(output)

        # concatenate outputs
        output = operator_packed(lambda seq: torch.cat(seq, dim=2), outputs)
        # same as torch.cat(outputs) except it supports PackedSequence

        # batch_size, n_samples, (sum of LSTM dimensions)

        if self.weighted:
            func = lambda b: b * self.alphas_
            output = map_packed(func, output)
            # same as output * self.alphas_ except it supports PackedSequence

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):
            func = lambda b: torch.tanh(layer(b))
            output = map_packed(func, output)
            # same as torch.tanh(layer(output)) except it supports PackedSequence

        # n_samples, batch_size, dimension

        if self.attention_layers_:

            attn = sequences
            for layer, hidden_dim in zip(self.attention_layers_,
                                         self.attention + [1]):
                func = lambda b: torch.tanh(layer(b))
                attn = map_packed(func, attn)
                # same as torch.tanh(layer(attn)) except it supports PackedSequence

            func = lambda oa: oa[0] * F.softmax(oa[1], dim=1)
            output = operator_packed(func, (output, attn))
            # same as output * F.softmax(attn, dim=1) except it supports PackedSequence

        # temporal pooling
        if self.pooling == 'sum':
            pool_func = lambda batch: batch.sum(dim=1)
            output = pool_packed(pool_func, output)
            # same as output.sum(dim=1) except it supports PackedSequence

        elif self.pooling == 'max':
            pool_func = lambda batch: batch.max(dim=1)[0]
            output = pool_packed(pool_func, output)
            # same as output.max(dim=1)[0] except it supports PackedSequence

        # batch_size, dimension

        # batch normalization
        if self.batch_normalize:
            output = self.batch_norm_(output)

        if self.normalize:
            norm = torch.norm(output, 2, 1, keepdim=True)

        if self.normalize == 'sphere':
            output = output / norm

        elif self.normalize == 'ball':
            output = output / norm * torch.sigmoid(self.norm_batch_norm_(norm))

        elif self.normalize == 'ring':
            norm_ = self.norm_batch_norm_(norm)
            output = output / norm * (1 + torch.sigmoid(self.norm_batch_norm_(norm)))

        # batch_size, dimension

        return output
