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


# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# Update TristouNet to latest API. Commenting it out for now.
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils.rnn import PackedSequence
# from torch.nn.utils.rnn import pad_packed_sequence
#
#
# class TristouNet(nn.Module):
#     """TristouNet sequence embedding
#
#     RNN ( » ... » RNN ) » temporal pooling › ( MLP › ... › ) MLP › normalize
#
#     Parameters
#     ----------
#     specifications : `dict`
#         Batch specifications:
#             {'X': {'dimension': n_features}}
#     rnn : {'LSTM', 'GRU'}, optional
#         Defaults to 'LSTM'.
#     recurrent: list, optional
#         List of output dimension of stacked RNNs.
#         Defaults to [16, ] (i.e. one RNN with output dimension 16)
#     bidirectional : bool, optional
#         Use bidirectional recurrent layers. Defaults to False.
#     pooling : {'sum', 'max'}
#         Temporal pooling strategy. Defaults to 'sum'.
#     linear : list, optional
#         List of hidden dimensions of linear layers. Defaults to [16, 16].
#
#     Reference
#     ---------
#     Hervé Bredin. "TristouNet: Triplet Loss for Speaker Turn Embedding."
#     ICASSP 2017 (https://arxiv.org/abs/1609.04301)
#     """
#
#     supports_packed = True
#
#     def __init__(self, specifications,
#                  rnn='LSTM', recurrent=[16], bidirectional=False,
#                  pooling='sum', linear=[16, 16]):
#
#         super(TristouNet, self).__init__()
#
#         self.specifications = specifications
#         self.n_features_ = specifications['X']['dimension']
#         self.rnn = rnn
#         self.recurrent = recurrent
#         self.bidirectional = bidirectional
#         self.pooling = pooling
#         self.linear = [] if linear is None else linear
#
#         self.num_directions_ = 2 if self.bidirectional else 1
#
#         if self.pooling not in {'sum', 'max'}:
#             raise ValueError('"pooling" must be one of {"sum", "max"}')
#
#         # create list of recurrent layers
#         self.recurrent_layers_ = []
#         input_dim = self.n_features_
#         for i, hidden_dim in enumerate(self.recurrent):
#             if self.rnn == 'LSTM':
#                 recurrent_layer = nn.LSTM(input_dim, hidden_dim,
#                                           bidirectional=self.bidirectional,
#                                           batch_first=True)
#             elif self.rnn == 'GRU':
#                 recurrent_layer = nn.GRU(input_dim, hidden_dim,
#                                          bidirectional=self.bidirectional,
#                                          batch_first=True)
#             else:
#                 raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
#             self.add_module('recurrent_{0}'.format(i), recurrent_layer)
#             self.recurrent_layers_.append(recurrent_layer)
#             input_dim = hidden_dim * (2 if self.bidirectional else 1)
#
#         # create list of linear layers
#         self.linear_layers_ = []
#         for i, hidden_dim in enumerate(self.linear):
#             linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
#             self.add_module('linear_{0}'.format(i), linear_layer)
#             self.linear_layers_.append(linear_layer)
#             input_dim = hidden_dim
#
#     @property
#     def dimension(self):
#         if self.linear:
#             return self.linear[-1]
#         return self.recurrent[-1] * (2 if self.bidirectional else 1)
#
#     def forward(self, sequence):
#         """
#
#         Parameters
#         ----------
#         sequence : (batch_size, n_samples, n_features) torch.Tensor
#
#         """
#
#         packed_sequences = isinstance(sequence, PackedSequence)
#
#         if packed_sequences:
#             _, n_features = sequence.data.size()
#             batch_size = sequence.batch_sizes[0].item()
#             device = sequence.data.device
#         else:
#             # check input feature dimension
#             batch_size, _, n_features = sequence.size()
#             device = sequence.device
#
#         if n_features != self.n_features_:
#             msg = 'Wrong feature dimension. Found {0}, should be {1}'
#             raise ValueError(msg.format(n_features, self.n_features_))
#
#         output = sequence
#
#         # recurrent layers
#         for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):
#
#             if self.rnn == 'LSTM':
#                 # initial hidden and cell states
#                 h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
#                                 device=device, requires_grad=False)
#                 c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
#                                 device=device, requires_grad=False)
#                 hidden = (h, c)
#
#             elif self.rnn == 'GRU':
#                 # initial hidden state
#                 hidden = torch.zeros(
#                     self.num_directions_, batch_size, hidden_dim,
#                     device=device, requires_grad=False)
#
#             # apply current recurrent layer and get output sequence
#             output, _ = layer(output, hidden)
#
#         if packed_sequences:
#             output, lengths = pad_packed_sequence(output, batch_first=True)
#
#         # batch_size, n_samples, dimension
#
#         # average temporal pooling
#         if self.pooling == 'sum':
#             output = output.sum(dim=1)
#         elif self.pooling == 'max':
#             if packed_sequences:
#                 msg = ('"max" pooling is not yet implemented '
#                        'for variable length sequences.')
#                 raise NotImplementedError(msg)
#             output, _ = output.max(dim=1)
#
#         # batch_size, dimension
#
#         # stack linear layers
#         for hidden_dim, layer in zip(self.linear, self.linear_layers_):
#
#             # apply current linear layer
#             output = layer(output)
#
#             # apply non-linear activation function
#             output = torch.tanh(output)
#
#         # batch_size, dimension
#
#         # unit-normalize
#         norm = torch.norm(output, 2, 1, keepdim=True)
#         output = output / norm
#
#         return output
