#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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

from . import TASK_MULTI_CLASS_CLASSIFICATION
from . import TASK_MULTI_LABEL_CLASSIFICATION
from . import TASK_REGRESSION
from . import TASK_REPRESENTATION_LEARNING

from .sincnet import SincNet


class RNN(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(self, n_features, unit='LSTM', hidden_size=16, num_layers=1,
                 bias=True, dropout=0, bidirectional=False, concatenate=False,
                 pool=None):
        super().__init__()

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.concatenate = concatenate

        if self.concatenate:

            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(input_dim, self.hidden_size,
                            num_layers=1, bias=self.bias,
                            batch_first=True, dropout=dropout,
                            bidirectional=self.bidirectional)

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(self.n_features, self.hidden_size,
                              num_layers=self.num_layers, bias=self.bias,
                              batch_first=True, dropout=self.dropout,
                              bidirectional=self.bidirectional)

        self.pool = pool

    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)

        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.

        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """

        if return_intermediate:
            num_directions = 2 if self.bidirectional else 1

        if self.concatenate:

            if return_intermediate:
                msg = (
                    '"return_intermediate" is not supported '
                    'when "concatenate" is True'
                )
                raise NotADirectoryError(msg)

            outputs = []

            # apply each layer separately...
            for i, rnn in enumerate(self.rnn_):
                if i > 0:
                    output, hidden = rnn(output, hidden)
                else:
                    output, hidden = rnn(features)
                outputs.append(output)

            # ... and concatenate their output
            output = torch.cat(outputs, dim=2)

        else:
            output, hidden = self.rnn_(features)

            if return_intermediate:
                if self.unit == 'LSTM':
                    h = hidden[0]
                elif self.unit == 'GRU':
                    h = hidden

                # to (num_layers, batch_size, num_directions * hidden_size)
                h = h.view(
                    self.num_layers, num_directions, -1, self.hidden_size)
                intermediate = h.transpose(2, 1).contiguous().view(
                    self.num_layers, -1, num_directions * self.hidden_size)

        if self.pool == 'sum':
            output = output.sum(dim=1)

        elif self.pool == 'max':
            output = output.max(dim=1)[0]

        elif self.pool == 'last':
            if self.bidirectional:
                output = torch.cat(
                    hidden.view(self.num_layers, num_directions,
                                -1, self.hidden_size)[-1],
                    dim=0)
            else:
                output = output[:, -1]

        elif self.pool == 'x-vector':
            output = torch.cat((torch.mean(output, dim=1),
                                torch.std(output, dim=1)),
                               dim=1)

        if return_intermediate:
            return output, intermediate

        return output

    def dimension():
        doc = "Output features dimension."
        def fget(self):
            dimension = self.hidden_size
            if self.bidirectional:
                dimension *= 2
            if self.concatenate:
                dimension *= self.num_layers
            if self.pool == 'x-vector':
                dimension *= 2
            return dimension
        return locals()
    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        dimension = self.hidden_size
        if self.bidirectional:
            dimension *= 2
        return dimension


class FF(nn.Module):
    """Feedforward layers

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    hidden_size : `list` of `int`, optional
        Linear layers hidden dimensions. Defaults to [16, ].
    """

    def __init__(self, n_features, hidden_size=[16, ]):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.linear_ = nn.ModuleList([])
        for hidden_size in self.hidden_size:
            linear = nn.Linear(n_features, hidden_size, bias=True)
            self.linear_.append(linear)
            n_features = hidden_size

    def forward(self, features):
        """

        Parameters
        ----------
        features : `torch.Tensor`
            (batch_size, n_samples, n_features) or (batch_size, n_features)

        Returns
        -------
        output : `torch.Tensor`
            (batch_size, n_samples, hidden_size[-1]) or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear_:
            output = linear(output)
            output = torch.tanh(output)
        return output

    def dimension():
        doc = "Output dimension."
        def fget(self):
            if self.hidden_size:
                return self.hidden_size[-1]
            return self.n_features
        return locals()
    dimension = property(**dimension())


class Embedding(nn.Module):
    """Embedding

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    unit_normalize : `boolean`, optional
        Normalize embeddings. Defaults to False.
    """

    def __init__(self, n_features, batch_normalize=False, unit_normalize=False):
        super().__init__()

        self.n_features = n_features
        # batch normalization ~= embeddings whitening.

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False)

        self.unit_normalize = unit_normalize

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        if self.unit_normalize:
            norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
            embedding = embedding / norm

        return embedding

    def dimension():
        doc = "Output dimension."
        def fget(self):
            return self.n_features
        return locals()
    dimension = property(**dimension())



class PyanNet(nn.Module):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    frame_crop = SincNet.frame_crop

    supports_packed = False

    @staticmethod
    def get_frame_info(sincnet=None, **kwargs):
        """
        """

        if sincnet is None:
            sincnet = dict()

        if sincnet.get('skip', False):
            return None

        return SincNet.get_frame_info(**sincnet)

    def __init__(self, specifications, sincnet=None, rnn=None, ff=None,
                 embedding=None):
        super().__init__()

        self.specifications = specifications
        self.task_ = specifications['task']

        n_features = specifications['X']['dimension']

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if sincnet.get('skip', False):
            pass
        else:
            if n_features != 1:
                msg = (
                    f'SincNet only supports mono waveforms. '
                    f'Here, waveform has {n_features} channels.'
                )
                raise ValueError(msg)

            self.sincnet_ = SincNet(**sincnet)
            self.frame_info_ = self.sincnet_.get_frame_info(**sincnet)
            n_features = self.sincnet_.dimension

        if rnn is None:
            rnn = dict()
        self.rnn = rnn
        self.rnn_ = RNN(n_features, **rnn)
        n_features = self.rnn_.dimension

        if ff is None:
            ff = dict()
        self.ff = ff
        self.ff_ = FF(n_features, **ff)
        n_features = self.ff_.dimension

        if self.task_ == TASK_REPRESENTATION_LEARNING:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return

        n_classes = len(specifications['y']['classes'])
        self.linear_ = nn.Linear(n_features, n_classes, bias=True)

        if self.task_ == TASK_MULTI_CLASS_CLASSIFICATION:
            self.activation_ = nn.LogSoftmax(dim=-1)

        elif self.task_ == TASK_MULTI_LABEL_CLASSIFICATION:
            self.activation_ = nn.Sigmoid()

        elif self.task_ == TASK_REGRESSION:
            self.activation_ = lambda x: x

        else:
            msg = f'Unsupported task type: {self.task_}'
            raise NotImplementedError(msg)

    def forward(self, waveforms, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """

        if self.sincnet.get('skip', False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        if return_intermediate is None or return_intermediate == 0:
            intermediate = output
            output = self.rnn_(output)
        else:
            return_intermediate -= 1
            # get RNN final AND intermediate outputs
            output, intermediate = self.rnn_(output, return_intermediate=True)
            # only keep hidden state of requested layer
            intermediate = intermediate[return_intermediate]

        output = self.ff_(output)

        if self.task_ == TASK_REPRESENTATION_LEARNING:
            return self.embedding_(output)

        output = self.linear_(output)
        output = self.activation_(output)

        if return_intermediate is None:
            return output
        return output, intermediate

    @property
    def dimension(self):
        if self.task_ == TASK_REPRESENTATION_LEARNING:
            return self.embedding_.dimension
        msg = (
            "Only representation learning models "
            "have a 'dimension' attribute."
        )
        raise NotImplementedError(msg)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer-1)

    @property
    def classes(self):
        if self.task_ != TASK_REPRESENTATION_LEARNING:
            return self.specifications['y']['classes']
        msg = (
            "Representation learning models "
            "do not have a 'classes' attribute."
        )
        raise NotImplementedError(msg)

    @property
    def n_classes(self):
        if self.task_ != TASK_REPRESENTATION_LEARNING:
            return len(self.specifications['y']['classes'])
        msg = (
            "Representation learning models "
            "do not have a 'n_classes' attribute."
        )
        raise NotImplementedError(msg)


class ClopiNet(PyanNet):

    def __init__(self, specifications):

        rnn = {
            'unit': 'LSTM',
            'hidden_size': 256,
            'num_layers': 3,
            'bidirectional': True,
            'concatenate': True,
            'pool': 'sum',
        }

        ff = {
            'hidden_size': [256, ],
        }

        embedding = {
            'batch_normalize': True,
            'unit_normalize': False,
        }

        super().__init__(specifications,
                         rnn=rnn,
                         ff=ff,
                         embedding=embedding)
