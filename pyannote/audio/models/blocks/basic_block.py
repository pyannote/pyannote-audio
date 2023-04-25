# The MIT License (MIT)
#
# Copyright (c) 2019-2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Clément Pagés

import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ A residual block composed of two convolutional layers and one residual
    connection linking the block input to the block output
    """
    expansion = 1

    def __init__(self, inplanes: int, outplanes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.shorcut = nn.Sequential()
        # Only in the case where dimensions in input and output of this block are
        # different, in order to match these dimensions:
        if stride != 1 or inplanes != self.expansion * outplanes:
            # Convolution layer only used to match dimension between input and
            # output of this block
            self.shorcut = nn.Sequential(nn.Conv2d(inplanes,
                                            self.expansion * outplanes,
                                            kernel_size=1,
                                            stride=stride,
                                            bias=False),
                                            nn.BatchNorm2d(self.expansion * outplanes))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_data : torch.Tensor
            batch of features with size (batch, channel, sample)
        """
        outputs = F.relu(self.bn1(self.conv1(input_data)))
        outputs = F.relu(self.bn2(self.conv2(outputs)))
        outputs += self.shorcut(input_data)
        outputs = F.relu(outputs)
        return outputs
