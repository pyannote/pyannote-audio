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
# Juan Manuel Coria

from typing_extensions import Literal
from warnings import warn

import torch
import torch.nn as nn


class TemporalPooling(nn.Module):
    """Pooling strategy over temporal sequences."""

    @staticmethod
    def create(method: Literal['sum', 'max', 'last', 'stats']) -> nn.Module:
        """Pooling strategy factory. returns an instance of `TemporalPooling` given its name.

        Parameters
        ----------
        method : {'sum', 'max', 'last', 'stats', 'x-vector'}
            Temporal pooling strategy. The `x-vector` method name
            for stats pooling (equivalent to `stats`) is kept for
            retrocompatibility but it will be removed in a future version.
        Returns
        -------
        output : nn.Module
            The temporal pooling strategy object
        """
        if method == 'sum':
            klass = SumPool
        elif method == 'max':
            klass = MaxPool
        elif method == 'last':
            klass = LastPool
        elif method == 'stats':
            klass = StatsPool
        elif method == 'x-vector':
            klass = StatsPool
            warn("`x-vector` is deprecated and will be removed in a future version. Please use `stats` instead")
        else:
            raise ValueError(f"`{method}` is not a valid temporal pooling method")
        return klass()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TemporalPooling subclass must implement `forward`")


class SumPool(TemporalPooling):
    """Calculate pooling as the sum over a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x.sum(dim=1)


class MaxPool(TemporalPooling):
    """Calculate pooling as the maximum over a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x.max(dim=1)[0]


class LastPool(TemporalPooling):
    """Calculate pooling as the last element of a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, hidden_size)
        """
        return x[:, -1]


class StatsPool(TemporalPooling):
    """Calculate pooling as the concatenated mean and standard deviation of a sequence"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, hidden_size)
            A batch of sequences.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, 2 * hidden_size)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)
