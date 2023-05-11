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


class SEBlock(nn.Module):
    """Implementation of a SE block from `Squeeze-and-Excitation Networks`
    https://arxiv.org/abs/1709.01507v4
    """
    def __init__(self, channel, reduction=16) -> None:
        super().__init__()
        # squeeze dimension = (batch_size, num_channel, 1, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """"""
        batch_size, num_channel, _, _ = x.size()
        z = self.squeeze(x).view(batch_size, num_channel)
        s = self.excitation(z).view(batch_size, num_channel, 1, 1)
        return x * s.expand_as(x)