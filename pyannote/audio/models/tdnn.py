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
# Juan Manuel Coria

# The TDNN class defined here was taken from https://github.com/jonasvdd/TDNN

# Please give proper credit to the authors if you are using TDNN based or X-Vector based
# models by citing their papers:

# Waibel, Alexander H., Toshiyuki Hanazawa, Geoffrey E. Hinton, Kiyohiro Shikano and Kevin J. Lang.
# "Phoneme recognition using time-delay neural networks."
# IEEE Trans. Acoustics, Speech, and Signal Processing 37 (1989): 328-339.
# https://pdfs.semanticscholar.org/cd62/c9976534a6a2096a38244f6cbb03635a127e.pdf?_ga=2.86820248.1800960571.1579515113-23298545.1575886658

# Peddinti, Vijayaditya, Daniel Povey and Sanjeev Khudanpur.
# "A time delay neural network architecture for efficient modeling of long temporal contexts."
# INTERSPEECH (2015).
# https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

# Snyder, David, Daniel Garcia-Romero, Gregory Sell, Daniel Povey and Sanjeev Khudanpur.
# "X-Vectors: Robust DNN Embeddings for Speaker Recognition."
# 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2018): 5329-5333.
# https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from .pooling import StatsPool


class TDNN(nn.Module):
    def __init__(
        self,
        context: list,
        input_channels: int,
        output_channels: int,
        full_context: bool = True,
    ):
        """
        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class

        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element

        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.

        :param context: The temporal context
        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param full_context: Indicates whether a full context needs to be used
        """
        super(TDNN, self).__init__()
        self.full_context = full_context
        self.input_dim = input_channels
        self.output_dim = output_channels

        context = sorted(context)
        self.check_valid_context(context, full_context)

        if full_context:
            kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
            self.temporal_conv = weight_norm(
                nn.Conv1d(input_channels, output_channels, kernel_size)
            )
        else:
            # use dilation
            delta = context[1] - context[0]
            self.temporal_conv = weight_norm(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=len(context),
                    dilation=delta,
                )
            )

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, sequence_length, input_channels]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, len(valid_steps), output_dim]
        """
        x = self.temporal_conv(torch.transpose(x, 1, 2))
        return F.relu(torch.transpose(x, 1, 2))

    @staticmethod
    def check_valid_context(context: list, full_context: bool) -> None:
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dil

        :param full_context: indicates whether the full context (dilation=1) will be used
        :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
        """
        if full_context:
            assert (
                len(context) <= 2
            ), "If the full context is given one must only define the smallest and largest"
            if len(context) == 2:
                assert context[0] + context[-1] == 0, "The context must be symmetric"
        else:
            assert len(context) % 2 != 0, "The context size must be odd"
            assert (
                context[len(context) // 2] == 0
            ), "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(
                    delta[0] == delta[i] for i in range(1, len(delta))
                ), "Intra context spacing must be equal!"


class XVectorNet(nn.Module):
    """
    X-Vector neural network architecture as defined by https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

    Parameters
    ----------
    input_dim : int, default 24
        dimension of the input frames
    embedding_dim : int, default 512
        dimension of latent embeddings
    """

    @property
    def dimension(self):
        return self.embedding_dim

    def __init__(self, input_dim: int = 24, embedding_dim: int = 512):
        super(XVectorNet, self).__init__()
        frame1 = TDNN(
            context=[-2, 2],
            input_channels=input_dim,
            output_channels=512,
            full_context=True,
        )
        frame2 = TDNN(
            context=[-2, 0, 2],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        frame3 = TDNN(
            context=[-3, 0, 3],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        frame4 = TDNN(
            context=[0], input_channels=512, output_channels=512, full_context=True
        )
        frame5 = TDNN(
            context=[0], input_channels=512, output_channels=1500, full_context=True
        )
        self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4, frame5, StatsPool())
        self.segment6 = nn.Linear(3000, embedding_dim)
        self.segment7 = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_intermediate: Optional[str] = None):
        """Calculate X-Vector network activations.
           Return the requested intermediate layer without computing unnecessary activations.

        Parameters
        ----------
        x : (batch_size, n_frames, out_channels)
            Batch of frames
        return_intermediate : 'stats_pool' | 'segment6' | 'segment7' | None
            If specified, return the activation of this specific layer.
            segment6 and segment7 activations are returned before the application of non linearity.

        Returns
        -------
        activations :
            (batch_size, 3000)               if return_intermediate == 'stats_pool'
            (batch_size, embedding_dim)      if return_intermediate == 'segment6' | 'segment7' | None
        """

        x = self.tdnn(x)

        if return_intermediate == "stats_pool":
            return x

        x = self.segment6(x)

        if return_intermediate == "segment6":
            return x

        x = self.segment7(F.relu(x))

        if return_intermediate == "segment7":
            return x

        return F.relu(x)
