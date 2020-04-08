# The MIT License (MIT)
#
# Copyright (c) 2019 Mirco Ravanelli
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
# Hervé Bredin - http://herve.niderb.fr

# Part of this code was taken from https://github.com/mravanelli/SincNet
# (see above license terms).

# Please give proper credit to the authors if you are using SincNet-based
# models  by citing their paper:

# Mirco Ravanelli, Yoshua Bengio.
# "Speaker Recognition from raw waveform with SincNet".
# SLT 2018. https://arxiv.org/abs/1808.00158

from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from pyannote.core import SlidingWindow
from pyannote.audio.train.task import Task


class SincConv1d(nn.Module):
    """Sinc-based 1D convolution

    Parameters
    ----------
    in_channels : `int`
        Should be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    stride : `int`, optional
        Defaults to 1.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    min_low_hz: `int`, optional
        Defaults to 50.
    min_band_hz: `int`, optional
        Defaults to 50.

    Usage
    -----
    Same as `torch.nn.Conv1d`

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio. "Speaker Recognition from raw waveform with
    SincNet". SLT 2018. https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):

        super().__init__()

        if in_channels != 1:
            msg = (
                f"SincConv1d only supports one input channel. "
                f"Here, in_channels = {in_channels}."
            )
            raise ValueError(msg)
        self.in_channels = in_channels

        self.out_channels = out_channels

        if kernel_size % 2 == 0:
            msg = (
                f"SincConv1d only support odd kernel size. "
                f"Here, kernel_size = {kernel_size}."
            )
            raise ValueError(msg)
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv1d does not support bias.")
        if groups > 1:
            raise ValueError("SincConv1d does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Half Hamming half window
        n_lin = torch.linspace(
            0, self.kernel_size / 2 - 1, steps=int((self.kernel_size / 2))
        )
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        # Due to symmetry, I only need half of the time axes
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        """Get sinc filters activations

        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent to Eq.4 of the reference paper
        # I just have expanded the sinc and simplified the terms.
        # This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincNet(nn.Module):
    """SincNet (learnable) feature extraction

    Parameters
    ----------
    waveform_normalize : `bool`, optional
        Standardize waveforms (to zero mean and unit standard deviation) and
        apply (learnable) affine transform. Defaults to True.
    instance_normalize : `bool`, optional
        Standardize internal representation (to zero mean and unit standard
        deviation) and apply (learnable) affine transform. Defaults to True.


    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio. "Speaker Recognition from raw waveform with
    SincNet". SLT 2018. https://arxiv.org/abs/1808.00158

    """

    @staticmethod
    def get_alignment(task: Task, **kwargs):
        """Get frame alignment"""
        return "strict"

    @staticmethod
    def get_resolution(
        task: Task,
        sample_rate: int = 16000,
        kernel_size: List[int] = [251, 5, 5],
        stride: List[int] = [1, 1, 1],
        max_pool: List[int] = [3, 3, 3],
        **kwargs,
    ) -> SlidingWindow:
        """Get frame resolution

        Parameters
        ----------
        task : Task
        sample_rate : int, optional
        kerne_size : list of int, optional
        stride : list of int, optional
        max_pool : list of int, optional

        Returns
        -------
        resolution : SlidingWindow
            Frame resolution.
        """

        # https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        padding = 0
        receptive_field, jump, start = 1, 1, 0.5
        for ks, s, mp in zip(kernel_size, stride, max_pool):
            # increase due to (Sinc)Conv1d
            receptive_field += (ks - 1) * jump
            start += ((ks - 1) / 2 - padding) * jump
            jump *= s
            # increase in receptive field due to MaxPool1d
            receptive_field += (mp - 1) * jump
            start += ((mp - 1) / 2 - padding) * jump
            jump *= mp

        return SlidingWindow(
            duration=receptive_field / sample_rate, step=jump / sample_rate, start=0.0
        )

    def __init__(
        self,
        waveform_normalize=True,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
        out_channels=[80, 60, 60],
        kernel_size: List[int] = [251, 5, 5],
        stride=[1, 1, 1],
        max_pool=[3, 3, 3],
        instance_normalize=True,
        activation="leaky_relu",
        dropout=0.0,
    ):
        super().__init__()

        # check parameters values
        n_layers = len(out_channels)
        if len(kernel_size) != n_layers:
            msg = (
                f"out_channels ({len(out_channels):d}) and kernel_size "
                f"({len(kernel_size):d}) should have the same length."
            )
            raise ValueError(msg)
        if len(stride) != n_layers:
            msg = (
                f"out_channels ({len(out_channels):d}) and stride "
                f"({len(stride):d}) should have the same length."
            )
            raise ValueError(msg)
        if len(max_pool) != n_layers:
            msg = (
                f"out_channels ({len(out_channels):d}) and max_pool "
                f"({len(max_pool):d}) should have the same length."
            )
            raise ValueError(msg)

        # Waveform normalization
        self.waveform_normalize = waveform_normalize
        if self.waveform_normalize:
            self.waveform_normalize_ = torch.nn.InstanceNorm1d(1, affine=True)

        # SincNet-specific parameters
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Conv1D parameters
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1d_ = nn.ModuleList([])

        # Max-pooling parameters
        self.max_pool = max_pool
        self.max_pool1d_ = nn.ModuleList([])

        # Instance normalization
        self.instance_normalize = instance_normalize
        if self.instance_normalize:
            self.instance_norm1d_ = nn.ModuleList([])

        config = zip(self.out_channels, self.kernel_size, self.stride, self.max_pool)

        in_channels = None
        for i, (out_channels, kernel_size, stride, max_pool) in enumerate(config):

            # 1D convolution
            if i > 0:
                conv1d = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=True,
                )
            else:
                conv1d = SincConv1d(
                    1,
                    out_channels,
                    kernel_size,
                    sample_rate=self.sample_rate,
                    min_low_hz=self.min_low_hz,
                    min_band_hz=self.min_band_hz,
                    stride=stride,
                    padding=0,
                    dilation=1,
                    bias=False,
                    groups=1,
                )
            self.conv1d_.append(conv1d)

            # 1D max-pooling
            max_pool1d = nn.MaxPool1d(max_pool, stride=max_pool, padding=0, dilation=1)
            self.max_pool1d_.append(max_pool1d)

            # 1D instance normalization
            if self.instance_normalize:
                instance_norm1d = nn.InstanceNorm1d(out_channels, affine=True)
                self.instance_norm1d_.append(instance_norm1d)

            in_channels = out_channels

        # Activation function
        self.activation = activation
        if self.activation == "leaky_relu":
            self.activation_ = nn.LeakyReLU(negative_slope=0.2)
        else:
            msg = f'Only "leaky_relu" activation is supported.'
            raise ValueError(msg)

        # Dropout
        self.dropout = dropout
        if self.dropout:
            self.dropout_ = nn.Dropout(p=self.dropout)

    def forward(self, waveforms):
        """Extract SincNet features

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1)
            Batch of waveforms

        Returns
        -------
        features : (batch_size, n_frames, out_channels[-1])
        """

        output = waveforms.transpose(1, 2)

        # standardize waveforms
        if self.waveform_normalize:
            output = self.waveform_normalize_(output)

        layers = zip(self.conv1d_, self.max_pool1d_)
        for i, (conv1d, max_pool1d) in enumerate(layers):

            output = conv1d(output)
            if i == 0:
                output = torch.abs(output)

            output = max_pool1d(output)

            if self.instance_normalize:
                output = self.instance_norm1d_[i](output)

            output = self.activation_(output)

            if self.dropout:
                output = self.dropout_(output)

        return output.transpose(1, 2)

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            return self.out_channels[-1]

        return locals()

    dimension = property(**dimension())
