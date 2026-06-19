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
#       ResNet basice block (+fwSE)
#------------------------------------------

class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """
    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)

        self.activation = nn.ReLU()  # Assuming ReLU, modify as needed

    def forward(self, inputs):
        # [bs, C, F, T]
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        # Reshape and apply excitation
        x = x[:,None,:,None]
        x = inputs * x
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, inc, outc, num_freq, stride=1, se_channels=64, Gdiv=4, use_fwSE=False):
        super().__init__()
        # if inc//Gdiv==0:
        #     Gdiv = inc
        self.conv1 = nn.Conv2d(inc, inc if Gdiv is not None else outc, kernel_size=3, stride=stride, padding=1, bias=False, 
                               groups=inc//Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()
            
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False, groups=outc//Gdiv if Gdiv is not None else 1)
        
        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()
            
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
            
        if outc != inc:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc),
            )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(residual)
        out = self.relu(out)
        # print(out.size())
        return out
