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

from pyannote.audio.models.blocks.se_block import SEBlock


class Bottleneck(nn.Module):
    """ A residual block composed of three convolutional layers and one residual
    connection linking the input of this block to its output
    """
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, se : bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, self.expansion *
                               outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * outplanes)
        self.se_block = SEBlock(outplanes)
        self.use_se = se

        self.shortcut = nn.Sequential()
        # Si une des dimensions est modififiées:
        if stride != 1 or inplanes != self.expansion * outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * outplanes,
                          kernel_size=1,  stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outplanes)
            )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_data : torch.Tensor
            batch of features with size (batch, channel, sample)
        """

        outputs = F.relu(self.bn1(self.conv1(input_data)))
        outputs = F.relu(self.bn2(self.conv2(outputs)))
        outputs = F.relu(self.bn3(self.conv3(outputs)))
        if self.use_se:
            outputs = self.se_block(outputs)
        outputs = outputs + self.shortcut(input_data)
        outputs = F.relu(outputs)
        return outputs
