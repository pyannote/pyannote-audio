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
from typing import List


class Linear(nn.Module):
    """Linear layers

    Parameters
    ----------
    n_features : int
        Input feature shape.
    hidden_size : list of int, optional
        Number of features in hidden. Defaults to [256, 128].
    bias : bool, optional
        If set to False, the layer will not learn an additive bias.
    """

    def __init__(
        self, n_features: int, hidden_size: List[int] = [256, 128], bias: bool = True,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.bias = bias

        self.linear = nn.ModuleList()

        for out_features in hidden_size:
            linear = nn.Linear(n_features, out_features, bias=self.bias)
            self.linear.append(linear)
            n_features = out_features

        self.activation = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameter
        ---------
        features : torch.Tensor
            Feature tensor with shape (batch_size, n_frames, n_features) or
            (batch_size, n_features).

        Returns
        -------
        output : torch.Tensor
            Output features with shape (batch_size, n_frames, hidden_size[-1])
            or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear:
            output = linear(output)
            output = self.activation(output)
        return output

    @property
    def num_layers(self) -> int:
        """Number of linear layers"""
        return len(self.hidden_size)

    @property
    def dimension(self) -> int:
        """Dimension of output features"""
        return self.hidden_size[-1]
