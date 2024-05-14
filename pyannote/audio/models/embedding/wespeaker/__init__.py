# MIT License
#
# Copyright (c) 2023 CNRS
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


from functools import lru_cache, partial
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

from .resnet import ResNet34, ResNet152, ResNet221, ResNet293


class BaseWeSpeakerResNet(Model):
    """Base class for WeSpeaker's ResNet models

    Parameters
    ----------
    fbank_centering_span : float, optional
        Span of the fbank centering window (in seconds).
        Defaults (None) to use whole input.

    See also
    --------
    torchaudio.compliance.kaldi.fbank

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: float = 25.0,  # in milliseconds
        frame_shift: float = 10.0,  # in milliseconds
        round_to_power_of_two: bool = True,
        snip_edges: bool = True,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        fbank_centering_span: Optional[float] = None,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.save_hyperparameters(
            "sample_rate",
            "num_channels",
            "num_mel_bins",
            "frame_length",
            "frame_shift",
            "dither",
            "round_to_power_of_two",
            "snip_edges",
            "window_type",
            "use_energy",
            "fbank_centering_span",
        )

        self._fbank = partial(
            kaldi.fbank,
            num_mel_bins=self.hparams.num_mel_bins,
            frame_length=self.hparams.frame_length,
            round_to_power_of_two=self.hparams.round_to_power_of_two,
            frame_shift=self.hparams.frame_shift,
            snip_edges=self.hparams.snip_edges,
            dither=self.hparams.dither,
            sample_frequency=self.hparams.sample_rate,
            window_type=self.hparams.window_type,
            use_energy=self.hparams.use_energy,
        )

    @property
    def fbank_only(self) -> bool:
        """Whether to only extract fbank features"""
        return getattr(self, "_fbank_only", False)

    @fbank_only.setter
    def fbank_only(self, value: bool):
        if hasattr(self, "receptive_field"):
            del self.receptive_field

        self._fbank_only = value

    def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)
            fbank features

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """

        waveforms = waveforms * (1 << 15)

        # fall back to CPU for FFT computation when using MPS
        # until FFT is fixed in MPS
        device = waveforms.device
        fft_device = torch.device("cpu") if device.type == "mps" else device

        features = torch.vmap(self._fbank)(waveforms.to(fft_device)).to(device)

        # center features with global average
        if self.hparams.fbank_centering_span is None:
            return features - torch.mean(features, dim=1, keepdim=True)

        # center features with running average
        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)
        kernel_size = conv1d_num_frames(
            num_samples=int(
                self.hparams.fbank_centering_span * self.hparams.sample_rate
            ),
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )
        return features - F.avg_pool1d(
            features.transpose(1, 2),
            kernel_size=2 * (kernel_size // 2) + 1,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        ).transpose(1, 2)

    @property
    def dimension(self) -> int:
        """Dimension of output"""

        if self.fbank_only:
            return self.hparams.num_mel_bins

        return self.resnet.embed_dim

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
        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)

        # TODO: take round_to_power_of_two and snip_edges into account

        num_frames = conv1d_num_frames(
            num_samples=num_samples,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )

        if self.fbank_only:
            return num_frames

        return self.resnet.num_frames(num_frames)

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

        receptive_field_size = num_frames

        if not self.fbank_only:
            receptive_field_size = self.resnet.receptive_field_size(
                receptive_field_size
            )

        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)

        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
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
        receptive_field_center = frame

        if not self.fbank_only:
            receptive_field_center = self.resnet.receptive_field_center(
                frame=receptive_field_center
            )

        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)
        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract speaker embeddings

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights passed to statistics pooling layer.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings.
        """

        fbank = self.compute_fbank(waveforms)
        if self.fbank_only:
            return fbank

        return self.resnet(fbank, weights=weights)[1]

    def forward_frames(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract frame-wise embeddings

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)

        Returns
        -------
        embeddings : (batch, ..., embedding_frames) torch.Tensor
            Batch of frame-wise embeddings.
        """
        fbank = self.compute_fbank(waveforms)
        return self.resnet.forward_frames(fbank)

    def forward_embedding(
        self, frames: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Extract speaker embeddings from frame-wise embeddings

        Parameters
        ----------
        frames : torch.Tensor
            Batch of frames with shape (batch, ..., embedding_frames).
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights passed to statistics pooling layer.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings.

        """
        return self.resnet.forward_embedding(frames, weights=weights)[1]

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract speaker embeddings

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights passed to statistics pooling layer.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings.
        """

        fbank = self.compute_fbank(waveforms)
        return self.resnet(fbank, weights=weights)[1]


class WeSpeakerResNet34(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet34(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet152(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet152(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet221(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet221(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet293(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet293(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


__all__ = [
    "WeSpeakerResNet34",
    "WeSpeakerResNet152",
    "WeSpeakerResNet221",
    "WeSpeakerResNet293",
]
