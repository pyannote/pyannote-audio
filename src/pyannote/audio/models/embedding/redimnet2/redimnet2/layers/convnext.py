# MIT License
# 
# Copyright (c) 2024 ID R&D, Inc.
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

import math
import torch
import functools
import numpy as np
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional

#------------------------------------------
#           ConvNeXtV2 block
#------------------------------------------

MaxPoolNd = {
    1 : nn.MaxPool1d,
    2 : nn.MaxPool2d
}

ConvNd = {
    1 : nn.Conv1d,
    2 : nn.Conv2d
}

BatchNormNd = {
    1 : nn.BatchNorm1d,
    2 : nn.BatchNorm2d
}

# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, C, dim=2, kernel_sizes=[(3,3),], Gdiv=1, padding='same', activation='gelu'):
        super().__init__()
        # if C//Gdiv==0:
        #     Gdiv = C
        self.dwconvs = nn.ModuleList(modules=[
            ConvNd[dim](C, C, kernel_size=ks, 
                                padding=padding, groups=C//Gdiv if Gdiv is not None else 1) for ks in kernel_sizes
        ])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1) # pointwise/1x1 convs, implemented with linear layers


    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs],dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x