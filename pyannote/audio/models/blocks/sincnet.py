# The MIT License (MIT)
#
# Copyright (c) 2019- CNRS
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
# Hervé Bredin - http://herve.niderb.fr

from functools import lru_cache
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB

from einops import rearrange

from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames,
    multi_conv_receptive_field_center,
    multi_conv_receptive_field_size,
)


class ConvBlock(nn.Module):
    """Multi-channels convolution block. Channels are processed together.
    Input shape: (batch, features, channels, frames)
    Output shape: (batch, features, frames)

    Parameters
    ----------
    num_channels: int
        number of channels in audio
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv2d = nn.Conv2d(80, 60, kernel_size=(num_channels, 5), stride=1)
        self.conv1d = nn.Conv1d(60, 60, kernel_size=5, stride=1)
        self.pool1d = nn.MaxPool1d(3, stride=3, padding=0, dilation=1)
        self.norm1d = nn.InstanceNorm1d(60, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.conv2d(x)
        outputs = torch.squeeze(outputs, dim=2)
        outputs = F.leaky_relu(self.norm1d(self.pool1d(outputs)))

        outputs = self.conv1d(outputs)
        outputs = F.leaky_relu(self.norm1d(self.pool1d(outputs)))

        return outputs


class SincNet(nn.Module):
    """Multi-channels SincNet

    Parameters
    ----------
    sample_rate: int, optional
        Audio sample rate. Only supports 16kHz for now.
    stride: int, optional
        Kernel stride. Defaults to 1.
    num_channels: int, optional
        Number of channels in audio. Defaults to 2.
    per_channel: bool, optional
        Whether to process audio's channel independently from each other.
        Defaults is False.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        stride: int = 1,
        num_channels: int = 1,
        channel_groups: Dict[str, List[int]] = None,
    ):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            # kernel_size by (sample_rate / 16000). but this needs to be double-checked.

        if not channel_groups:
            channel_groups = {
                str(channel_idx): [channel_idx] for channel_idx in range(num_channels)
            }

        self.sample_rate = sample_rate
        self.stride = stride
        self.num_channels = num_channels
        self.channel_groups = channel_groups

        self.wav_norm1d = nn.InstanceNorm1d(num_features=num_channels, affine=True)

        self.encoder = Encoder(
            ParamSincFB(
                80,
                251,
                stride=self.stride,
                sample_rate=sample_rate,
                min_low_hz=50,
                min_band_hz=50,
            )
        )

        self.pool2d = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1
        )
        self.norm2d = nn.InstanceNorm2d(80, affine=True)

        self.conv_blocks = nn.ModuleList()
        for _, channel_idxes in channel_groups.items():
            self.conv_blocks.append(ConvBlock(num_channels=len(channel_idxes)))

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

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
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

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
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

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """
        outputs = self.wav_norm1d(waveforms)

        outputs = self.encoder(outputs)
        # https://github.com/mravanelli/SincNet/issues/4
        outputs = torch.abs(outputs)
        if self.num_channels == 1:
            outputs = outputs.unsqueeze(dim=1)
        outputs = rearrange(outputs, "b c f t -> b f c t")
        outputs = F.leaky_relu(self.norm2d(self.pool2d(outputs)))

        conv_outputs = []
        for conv_block, channel_idxes in zip(
            self.conv_blocks, self.channel_groups.values()
        ):
            conv_outputs.append(conv_block(outputs[:, :, channel_idxes, :]))

        outputs = torch.stack(conv_outputs, dim=2)

        _, _, num_channels, _ = outputs.shape

        # needed for backward compatibility
        if num_channels == 1:
            outputs = torch.squeeze(outputs, dim=2)

        return outputs
