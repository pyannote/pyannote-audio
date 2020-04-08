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
from typing import List
from pyannote.core import SlidingWindow
from pyannote.audio.train.task import Task


class Convolutional(nn.Module):
    """Convolutional layers

    Parameters
    ----------
    n_features : int,
        Input feature shape. Should be 1.
    sample_rate: int, optional
        Input sample rate (in Hz). Defaults to 16000.
    out_channels : list of int, optional
        Number of channels produced by the convolutions.
    kernel_size : list of int, optional
        Size of the convolving kernels.
    stride : list of int, optional
        Stride of the convolutions
    max_pool : list of int, optional
        Size and stride of the size of the windows to take a max over.
    dropout : float, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
    """

    def __init__(
        self,
        n_features: int,
        sample_rate: int = 16000,
        out_channels: List[int] = [512, 512, 512, 512, 512, 512],
        kernel_size: List[int] = [251, 5, 5, 5, 5, 5],
        stride: List[int] = [5, 1, 1, 1, 1, 1],
        max_pool: List[int] = [3, 3, 3, 3, 3, 3],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_features = n_features
        self.sample_rate = sample_rate
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = max_pool
        self.dropout = dropout

        self.conv1ds = nn.ModuleList()
        self.pool1ds = nn.ModuleList()
        self.activation = nn.LeakyReLU(negative_slope=1e-2, inplace=False)
        if self.dropout > 0.0:
            self._dropout = nn.Dropout(p=self.dropout, inplace=False)

        in_channels = self.n_features
        for i, (out_channels, kernel_size, stride, max_pool) in enumerate(
            zip(self.out_channels, self.kernel_size, self.stride, self.max_pool)
        ):
            conv1d = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
            self.conv1ds.append(conv1d)

            pool1d = nn.MaxPool1d(
                max_pool,
                stride=max_pool,
                padding=0,
                dilation=1,
                return_indices=False,
                ceil_mode=False,
            )
            self.pool1ds.append(pool1d)

            in_channels = out_channels

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameter
        ---------
        waveform : torch.Tensor
            Waveform tensor with shape (batch_size, n_samples, 1)

        Returns
        -------
        features : torch.Tensor
            Output features (batch_size, n_frames, n_features)
        """

        output = waveforms.transpose(1, 2)

        for i, (conv1d, pool1d) in enumerate(zip(self.conv1ds, self.pool1ds)):
            output = conv1d(output)
            output = pool1d(output)
            output = self.activation(output)
            if self.dropout > 0.0 and i + 1 < self.num_layers:
                output = self._dropout(output)

        return output.transpose(1, 2)

    @property
    def num_layers(self):
        """Number of convolutional layers"""
        return len(self.out_channels)

    @property
    def dimension(self):
        """Dimension of output features"""
        return self.out_channels[-1]

    @staticmethod
    def get_alignment(task: Task, **kwargs) -> Text:
        """Output frame alignment"""
        return "strict"

    @staticmethod
    def get_resolution(
        task: Task,
        sample_rate: int = 16000,
        out_channels: List[int] = [512, 512, 512, 512, 512, 512],
        kernel_size: List[int] = [251, 5, 5, 5, 5, 5],
        stride: List[int] = [5, 1, 1, 1, 1, 1],
        max_pool: List[int] = [3, 3, 3, 3, 3, 3],
        **kwargs,
    ) -> SlidingWindow:
        """Output frame resolution"""

        # https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        padding = 0
        receptive_field, jump, start = 1, 1, 0.5
        for ks, s, mp in zip(kernel_size, stride, max_pool):
            # increase due to (Sinc)Conv1d
            receptive_field += (ks - 1) * jump
            start += ((ks - 1) / 2 - padding) * jump
            jump *= s
            # increase in receptive field due to MaxPool1d
            receptive_field += (mp - 1) * jump
            start += ((mp - 1) / 2 - padding) * jump
            jump *= mp

        return SlidingWindow(
            duration=receptive_field / sample_rate, step=jump / sample_rate, start=0.0
        )
