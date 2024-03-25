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

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class XVectorMFCC(Model):
    MFCC_DEFAULTS = {"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        mfcc: Optional[dict] = None,
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
        self.kernel_size = [5, 3, 3, 1, 1]
        self.dilation = [1, 2, 3, 1, 1]
        self.padding = [0, 0, 0, 0, 0]
        self.stride = [1, 1, 1, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, self.kernel_size, self.dilation
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

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.hparams.dimension

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        num_frames = conv1d_num_frames(
            num_samples,
            kernel_size=n_fft,
            stride=hop_length,
            dilation=1,
            padding=n_fft // 2 if center else 0,
        )

        return multi_conv_num_frames(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = multi_conv_receptive_field_size(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft

        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=n_fft,
            stride=hop_length,
            dilation=1,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        receptive_field_center = multi_conv_receptive_field_center(
            frame,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=n_fft,
            stride=hop_length,
            padding=n_fft // 2 if center else 0,
            dilation=1,
        )

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
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
        sincnet: Optional[dict] = None,
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
        self.kernel_size = [5, 3, 3, 1, 1]
        self.dilation = [1, 2, 3, 1, 1]
        self.padding = [0, 0, 0, 0, 0]
        self.stride = [1, 1, 1, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, self.kernel_size, self.dilation
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

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.hparams.dimension

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = self.sincnet.num_frames(num_samples)

        return multi_conv_num_frames(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = multi_conv_receptive_field_size(
            num_frames,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return self.sincnet.receptive_field_size(num_frames=receptive_field_size)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        receptive_field_center = multi_conv_receptive_field_center(
            frame,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        return self.sincnet.receptive_field_center(frame=receptive_field_center)

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
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
