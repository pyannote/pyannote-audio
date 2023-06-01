# MIT License
#
# Copyright (c) 2023- CNRS
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

import decimal
from math import sqrt
import torch

from pyannote.audio.models.blocks.pooling import StatsPool


def test_stats_pool():

    x = torch.Tensor([
        [[2., 4.]],
        [[1., 1.]]
    ])

    # 2D weights tensor
    w1 = torch.Tensor([
        [0.5, 0.01],
        [0.2, 0.1],
    ])

    # two speaker tensor
    w2 = torch.Tensor([
        [[0.1, 0.2], [0.2, 0.3]],
        [[0.001, 0.001], [0.2, 0.3]]
    ])

    stats_pool = StatsPool()

    y0 = stats_pool(x)
    y1 = stats_pool(x, w1)
    y2 = stats_pool(x, w2)

    assert(torch.equal(torch.round(y0, decimals=4), torch.Tensor([[3., 1.4142], [1., 0.]])))
    assert(torch.equal(torch.round(y1, decimals=4), torch.Tensor([[2.0392, 1.4142], [1., 0.]])))
    assert(torch.equal(torch.round(y2, decimals=4), torch.Tensor([[[3.3333, 1.4142], [3.2, 1.4142]], [[1.0, 0.], [1.0, 0.0]]])))







