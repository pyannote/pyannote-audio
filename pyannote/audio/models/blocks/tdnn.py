# MIT License
#
# Copyright (c) [year] Jonas Van Der Donckt
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class TDNN(nn.Module):
    def __init__(
        self,
        context: List[int],
        input_channels: int,
        output_channels: int,
        full_context: bool = True,
    ):
        """TDNN layer (taken from https://github.com/jonasvdd/TDNN)

        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class

        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element

        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.

        Parameters
        ----------
        context : List[int]
            Temporal context.
        input_channels : int
            Number of input channels
        output_channels : int
            Number of channels produced by the temporal convolution
        full_context : bool
            Indicates whether a full context needs to be used
        """

        super().__init__()
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

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x : (batch, channel, time) torch.Tensor

        Returns
        -------
        outputs : (batch, out_channel, time)
        """

        return F.relu(self.temporal_conv(x))

    @staticmethod
    def check_valid_context(context: List[int], full_context: bool) -> None:
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dilation

        Parameters
        ----------
        full_context : bool
            Indicates whether the full context (dilation=1) will be used
        context : list of int
            The context of the model, must be symmetric if no full context and have an equal spacing.
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
