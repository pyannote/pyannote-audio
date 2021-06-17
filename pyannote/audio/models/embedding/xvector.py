# MIT License
#
# Copyright (c) 2021 CNRS
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

from typing import Optional

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict


class XVectorMFCC(Model):

    MFCC_DEFAULTS = {"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        mfcc: dict = None,
        dimension: int = 512,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        mfcc = merge_dict(self.MFCC_DEFAULTS, mfcc)
        mfcc["sample_rate"] = sample_rate

        self.save_hyperparameters("mfcc", "dimension")

        self.mfcc = MFCC(**self.hparams.mfcc)

        self.tdnns = nn.ModuleList()
        in_channel = self.hparams.mfcc["n_mfcc"]
        out_channels = [512, 512, 512, 512, 1500]
        kernel_sizes = [5, 3, 3, 1, 1]
        dilations = [1, 2, 3, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, kernel_sizes, dilations
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()

        self.embedding = nn.Linear(in_channel * 2, self.hparams.dimension)

    def forward(
        self, waveforms: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        outputs = self.mfcc(waveforms).squeeze(dim=1)
        for block in self.tdnns:
            outputs = block(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)


class XVectorSincNet(Model):

    SINCNET_DEFAULTS = {"stride": 10}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sincnet: dict = None,
        dimension: int = 512,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        self.save_hyperparameters("sincnet", "dimension")

        self.sincnet = SincNet(**self.hparams.sincnet)
        in_channel = 60

        self.tdnns = nn.ModuleList()
        out_channels = [512, 512, 512, 512, 1500]
        kernel_sizes = [5, 3, 3, 1, 1]
        dilations = [1, 2, 3, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, kernel_sizes, dilations
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()

        self.embedding = nn.Linear(in_channel * 2, self.hparams.dimension)

    def forward(
        self, waveforms: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        outputs = self.sincnet(waveforms).squeeze(dim=1)
        for tdnn in self.tdnns:
            outputs = tdnn(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)
