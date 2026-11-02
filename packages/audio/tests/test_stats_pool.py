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

import torch

from pyannote.audio.models.blocks.pooling import StatsPool


def test_stats_pool_weightless():
    x = torch.Tensor([[[2.0, 4.0], [2.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # (batch = 2, features = 2, frames = 2)

    stats_pool = StatsPool()

    y = stats_pool(x)
    # (batch = 2, features = 4)

    assert torch.equal(
        torch.round(y, decimals=4),
        torch.Tensor([[3.0, 3.0, 1.4142, 1.4142], [1.0, 1.0, 0.0, 0.0]]),
    )


def test_stats_pool_one_speaker():
    x = torch.Tensor([[[2.0, 4.0], [2.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # (batch = 2, features = 2, frames = 2)

    w = torch.Tensor(
        [
            [0.5, 0.01],
            [0.2, 0.1],
        ]
    )
    # (batch = 2, frames = 2)

    stats_pool = StatsPool()

    y = stats_pool(x, weights=w)
    # (batch = 2, features = 4)

    assert torch.equal(
        torch.round(y, decimals=4),
        torch.Tensor([[2.0392, 2.0392, 1.4142, 1.4142], [1.0, 1.0, 0.0, 0.0]]),
    )


def test_stats_pool_multi_speaker():
    x = torch.Tensor([[[2.0, 4.0], [2.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # (batch = 2, features = 2, frames = 2)

    w = torch.Tensor([[[0.1, 0.2], [0.2, 0.3]], [[0.001, 0.001], [0.2, 0.3]]])
    # (batch = 2, speakers = 2, frames = 2)

    stats_pool = StatsPool()

    y = stats_pool(x, weights=w)
    # (batch = 2, speakers = 2, features = 4)

    assert torch.equal(
        torch.round(y, decimals=4),
        torch.Tensor(
            [
                [[3.3333, 3.3333, 1.4142, 1.4142], [3.2, 3.2, 1.4142, 1.4142]],
                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            ]
        ),
    )


def test_stats_pool_frame_mismatch():
    x = torch.Tensor([[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # (batch = 2, features = 2, frames = 2)

    stats_pool = StatsPool()
    w = torch.Tensor(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    # (batch = 2, frames = 3)

    y = stats_pool(x, weights=w)
    # (batch = 2, features = 4)

    assert torch.equal(
        torch.round(y, decimals=4),
        torch.Tensor([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
    )


def test_stats_pool_all_zero_weights():
    x = torch.Tensor([[[2.0, 4.0], [2.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]])
    # (batch = 2, features = 2, frames = 2)

    w = torch.Tensor(
        [
            [0.5, 0.01],
            [0.0, 0.0],  # all zero weights
        ]
    )
    # (batch = 2, frames = 2)

    stats_pool = StatsPool()

    y = stats_pool(x, weights=w)
    # (batch = 2, features = 4)

    assert torch.equal(
        torch.round(y, decimals=4),
        torch.Tensor([[2.0392, 2.0392, 1.4142, 1.4142], [0.0, 0.0, 0.0, 0.0]]),
    )
