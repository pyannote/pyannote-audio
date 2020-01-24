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


def get_pooling_strategy(name: Optional[str],
                         bidirectional: Optional[bool] = None,
                         num_layers: Optional[int] = None) -> Optional[nn.Module]:
    """Pooling strategy factory. returns an instance of a pooling module given its name

    Parameters
    ----------
    name : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy.
    bidirectional : `bool`, optional
        If True, assumes the input to come from a bidirectional RNN.
        It cannot be None if name == `last`. Defaults to None.
    num_layers : `int`, optional
        Number of recurrent layers if the input comes from a RNN
        It cannot be None if name == `last`. Defaults to None.
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
        return LastPool(bidirectional, num_layers)
    elif name == 'x-vector':
        return StatsPool()
    else:
        return None


class SumPool(nn.Module):

    def forward(self, x: torch.Tensor):
        """Calculate sum of the input frames

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames

        Returns
        -------
        output : (batch_size, out_channels)
        """
        return x.sum(dim=1)


class MaxPool(nn.Module):

    def forward(self, x: torch.Tensor):
        """Calculate maximum of the input frames

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames

        Returns
        -------
        output : (batch_size, out_channels)
        """
        return x.max(dim=1)[0]


class LastPool(nn.Module):
    """TODO

    Parameters
    ----------
    bidirectional : `boolean`, optional
        If True, the hidden outputs of a bidirectional RNN are assumed as the input. Usual output otherwise.
        Defaults to False.
    num_layers : `int`, optional
        Number of recurrent layers of the RNN. Defaults to 1.
    """

    def __init__(self, bidirectional: bool = False, num_layers: int = 1):
        super(LastPool, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        """TODO"""
        if self.bidirectional:
            return torch.cat(
                x.view(self.num_layers, self.num_directions,
                       -1, self.hidden_size)[-1],
                dim=0)
        else:
            return x[:, -1]


class StatsPool(nn.Module):

    def forward(self, x: torch.Tensor):
        """Calculate mean and standard deviation of the input frames and concatenate them

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames

        Returns
        -------
        output : (batch_size, 2 * out_channels)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)
