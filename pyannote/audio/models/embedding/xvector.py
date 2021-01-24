# MIT License
#
# Copyright (c) 2020 CNRS
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
import torch.nn.functional as F
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.tdnn import TDNN


class XVectorMFCC(Model):

    MFCC_DEFAULTS = {"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        mfcc: dict = None,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        mfcc_params = dict(**self.MFCC_DEFAULTS)
        if mfcc is not None:
            mfcc_params.update(**mfcc)
        mfcc_params["sample_rate"] = sample_rate
        self.hparams.mfcc = mfcc_params
        self.mfcc = MFCC(**self.hparams.mfcc)

        self.frame1 = TDNN(
            context=[-2, 2],
            input_channels=self.hparams.mfcc["n_mfcc"],
            output_channels=512,
            full_context=True,
        )
        self.frame2 = TDNN(
            context=[-2, 0, 2],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame3 = TDNN(
            context=[-3, 0, 3],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame4 = TDNN(
            context=[0], input_channels=512, output_channels=512, full_context=True
        )
        self.frame5 = TDNN(
            context=[0], input_channels=512, output_channels=1500, full_context=True
        )

        self.stats_pool = StatsPool()

        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)

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
        outputs = self.frame1(outputs)
        outputs = self.frame2(outputs)
        outputs = self.frame3(outputs)
        outputs = self.frame4(outputs)
        outputs = self.frame5(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        outputs = self.segment6(F.relu(outputs))
        return self.segment7(F.relu(outputs))


class XVector(Model):

    SINCNET_DEFAULTS = {"stride": 1}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sincnet: dict = None,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet_hparams = dict(**self.SINCNET_DEFAULTS)
        if sincnet is not None:
            sincnet_hparams.update(**sincnet)
        sincnet_hparams["sample_rate"] = sample_rate
        self.hparams.sincnet = sincnet_hparams
        self.sincnet = SincNet(**self.hparams.sincnet)

        self.frame1 = TDNN(
            context=[-2, 2],
            input_channels=60,
            output_channels=512,
            full_context=True,
        )
        self.frame2 = TDNN(
            context=[-2, 0, 2],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame3 = TDNN(
            context=[-3, 0, 3],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame4 = TDNN(
            context=[0], input_channels=512, output_channels=512, full_context=True
        )
        self.frame5 = TDNN(
            context=[0], input_channels=512, output_channels=1500, full_context=True
        )

        self.stats_pool = StatsPool()

        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)

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

        outputs = self.sincnet(waveforms)
        outputs = self.frame1(outputs)
        outputs = self.frame2(outputs)
        outputs = self.frame3(outputs)
        outputs = self.frame4(outputs)
        outputs = self.frame5(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        outputs = self.segment6(F.relu(outputs))
        return self.segment7(F.relu(outputs))
