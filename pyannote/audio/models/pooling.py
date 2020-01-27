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

from typing import Optional

import torch
import torch.nn as nn


def get_temporal_pooling(name: Optional[str],
                         bidirectional: Optional[bool] = None,
                         num_layers: Optional[int] = None,
                         hidden_size: Optional[int] = None) -> Optional[nn.Module]:
    """Pooling strategy factory. returns an instance of a pooling module given its name

    Parameters
    ----------
    name : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy.
    bidirectional : `bool`, optional
        If True, assumes the input to come from a bidirectional RNN.
        Mandatory if name == `last`. Defaults to None.
    num_layers : `int`, optional
        Number of recurrent layers of the RNN.
        Mandatory if name == `last`. Defaults to None.
    hidden_size : `int`, optional
        Number of features in the hidden state of the RNN. Defaults to 16.
        Mandatory if name == `last`. Defaults to None.
    Returns
    -------
    output : nn.Module or None if invalid name
        the pooling strategy
    """
    if name == 'sum':
        return SumPool()
    elif name == 'max':
        return MaxPool()
    elif name == 'last':
        return LastPool(bidirectional, num_layers, hidden_size)
    elif name == 'x-vector':
        return StatsPool()
    else:
        return None


class SumPool(nn.Module):

    def forward(self, hidden: Optional[torch.Tensor], x: torch.Tensor):
        """Calculate pooling as the sum over a RNN sequence.

        Parameters
        ----------
        hidden : unused, kept to respect common interface
        x : `torch.Tensor`, shape (seq_len, batch_size, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        return x.sum(dim=1)


class MaxPool(nn.Module):

    def forward(self, hidden: Optional[torch.Tensor], x: torch.Tensor):
        """Calculate pooling as the maximum values over a RNN sequence.

        Parameters
        ----------
        hidden : unused, kept to respect common interface
        x : `torch.Tensor`, shape (seq_len, batch_size, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        return x.max(dim=1)[0]


class LastPool(nn.Module):
    """Pooling strategy to keep the last activation of a RNN.

    Parameters
    ----------
    bidirectional : `boolean`
        If True, the hidden outputs of a bidirectional RNN are assumed as the input. Usual output otherwise.
        Defaults to False.
    num_layers : `int`
        Number of recurrent layers of the RNN. Defaults to 1.
    hidden_size : `int`
        Number of features in the hidden state of the RNN. Defaults to 16.
    """

    def __init__(self, bidirectional: bool = False, num_layers: int = 1, hidden_size: int = 16):
        super(LastPool, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

    def forward(self, hidden: torch.Tensor, x: torch.Tensor):
        """Return the last activation of a RNN sequence.

        Parameters
        ----------
        hidden : `torch.Tensor`, shape (num_layers * num_directions, batch_size, hidden_size)
            Hidden states of a RNN.
        x : `torch.Tensor`, shape (seq_len, batch_size, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        if self.bidirectional:
            return torch.cat(
                hidden.view(self.num_layers, self.num_directions, -1, self.hidden_size)[-1],
                dim=0)
        else:
            return x[:, -1]


class StatsPool(nn.Module):

    def forward(self, hidden, x):
        """Calculate mean and standard deviation of a RNN sequence and concatenate them.

        Parameters
        ----------
        hidden : unused, kept to respect common interface
        x : `torch.Tensor`, shape (seq_len, batch_size, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, 2 * num_directions * hidden_size)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)
