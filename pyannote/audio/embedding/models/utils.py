#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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


import math


def get_conv1d_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Predict output shape of Conv1D"""

    out_shape = input_shape + 2 * padding - dilation * (kernel_size - 1) - 1
    out_shape = out_shape / stride + 1
    return int(math.floor(out_shape))


def get_conv2d_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Predict output shape of Conv2D"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    h_in, w_in = input_shape

    h_out = h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    h_out = h_out / stride[0] + 1

    w_out = w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    w_out = w_out / stride[1] + 1

    return int(math.floor(h_out)), int(math.floor(w_out))
