#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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
from typing import Text


class Recurrent(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : int
        Input feature shape.
    unit : {"LSTM", "GRU"}, optional
        Defaults to "LSTM".
    hidden_size : int, optional
        Number of features in the hidden state h. Defaults to 512.
    num_layers : int, optional
        Number of recurrent layers. Defaults to 1.
    bias : bool, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : float, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : bool, optional
        Use bidirectional RNN. Defaults to True.
    probes : bool, optional
        Split multi-layer RNN into multiple one-layer RNNs to expose
        corresponding probes (see pyannote.audio.train.model.Model.probes).
        Might be useful when using a multi-layer RNN as the trunk of a larger
        multi-task RNN tree.
    """

    def __init__(
        self,
        n_features: int,
        unit: Text = "LSTM",
        hidden_size: int = 512,
        num_layers: int = 1,
        bias: int = True,
        dropout: float = 0.0,
        bidirectional: bool = True,
        probes: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.unit = unit
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.probes = probes

        if num_layers < 1:
            if bidirectional:
                msg = "'bidirectional' must be set to False when num_layers < 1"
                raise ValueError(msg)
            return

        Klass = getattr(nn, self.unit)

        if probes:

            self.rnn = nn.ModuleList([])

            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                dropout = 0 if (i + 1 == self.num_layers) else self.dropout

                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )

                self.rnn.append(rnn)

        else:
            self.rnn = Klass(
                self.n_features,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply recurrent layer

        Parameters
        ----------
        features : torch.Tensor
            Input feature sequence with shape (batch_size, n_frames, n_features).

        Returns
        -------
        output : torch.Tensor
            Output sequence with shape (batch_size, n_frames, hidden_size x n_directions).
        """

        if self.num_layers < 1:
            return features

        if self.probes:

            output, hidden = None, None
            for i, rnn in enumerate(self.rnn):

                rnn = getattr(self, f"rnn{i+1:02d}")

                if i > 0:
                    output, hidden = rnn(output, hidden)
                else:
                    output, hidden = rnn(features)

        else:
            output, hidden = self.rnn(features)

        return output

    @property
    def dimension(self):

        if self.num_layers < 1:
            return self.n_features

        dimension = self.hidden_size
        if self.bidirectional:
            dimension *= 2

        return dimension
