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

from typing import Optional

import torch
import torch.nn as nn


class TemporalPooling(nn.Module):
    """Pooling strategy over RNN sequences."""

    @staticmethod
    def create(name: Optional[str]) -> Optional[nn.Module]:
        """Pooling strategy factory. returns an instance of `TemporalPooling` given its name.

        Parameters
        ----------
        name : {'sum', 'max', 'last', 'x-vector'}, optional
            Temporal pooling strategy.
        Returns
        -------
        output : nn.Module or None if invalid name
            The temporal pooling strategy object
        """
        klass = None

        if name == 'sum':
            klass = SumPool
        elif name == 'max':
            klass = MaxPool
        elif name == 'last':
            klass = LastPool
        elif name == 'x-vector':
            klass = StatsPool

        return klass() if klass is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TemporalPooling subclass must implement `forward`")


class SumPool(TemporalPooling):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate pooling as the sum over a RNN sequence.

        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        return x.sum(dim=1)


class MaxPool(TemporalPooling):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate pooling as the maximum values over a RNN sequence.

        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        return x.max(dim=1)[0]


class LastPool(TemporalPooling):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last activation of a RNN sequence.

        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, num_directions * hidden_size)
        """
        if self.bidirectional:
            # TODO does this really make sense?
            raise NotImplementedError("LastPool does not yet support bidirectional RNNs")
        else:
            return x[:, -1]


class StatsPool(TemporalPooling):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate mean and standard deviation of a RNN sequence and concatenate them.

        Parameters
        ----------
        x : `torch.Tensor`, shape (batch_size, seq_len, num_directions * hidden_size)
            Output of a RNN.

        Returns
        -------
        output : `torch.Tensor`, shape (batch_size, 2 * num_directions * hidden_size)
        """
        mean, std = torch.mean(x, dim=1), torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)
