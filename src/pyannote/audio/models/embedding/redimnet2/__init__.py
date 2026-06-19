# MIT License
#
# Copyright (c) 2025 CNRS
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

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.receptive_field import (
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

from .redimnet2 import ReDimNet2Wrap


class BaseReDimNet2(Model):
    """Base class for ReDimNet2 speaker embedding models

    ReDimNet2 ("Scaling Speaker Verification via Time-Pooled Dimension Reshaping")
    extracts its own (log-mel) features from the waveform. As with the WeSpeaker
    models, these features are exposed through `compute_fbank` and, when
    `fbank_only` is set, returned directly by `forward`.

    Parameters
    ----------
    config : dict, optional
        Architecture configuration passed to `ReDimNet2Wrap`. This is exactly the
        `model_config` dictionary shipped alongside the official pretrained weights.
        Defaults (None) to an empty dictionary, i.e. `ReDimNet2Wrap` defaults.
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task addressed by the model.

    See also
    --------
    pyannote.audio.models.embedding.redimnet2.redimnet2.ReDimNet2Wrap
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        config = dict(config or {})
        self.save_hyperparameters("config")

        self.redimnet = ReDimNet2Wrap(**config)

        # (mel) feature extraction parameters, needed to map samples to frames.
        # `hop_length` is a `ReDimNet2Wrap` argument while `win_length` lives in
        # `spec_params` (passed through to the feature extractor). As in the
        # WeSpeaker models, the centering padding is ignored, so `num_frames` is a
        # (close) approximation -- this does not affect the `weights` path, which
        # interpolates weights to the actual number of feature frames.
        spec_params = config.get("spec_params", {})
        self._hop_length = config.get("hop_length", 160)
        self._win_length = spec_params.get("win_length", 400)
        # number of mel bins is the `F` (frequency) dimension fed to the backbone
        self._num_mel_bins = config.get("F", 72)

    @property
    def resnet(self):
        """Temporary alias for `redimnet` (WeSpeaker-style compatibility)

        Some code expects a WeSpeaker-style model exposing its inner network as
        `model.resnet` (e.g. to call `forward_frames` / `forward_embedding` on it).
        This alias lets such code run against ReDimNet2 unchanged. It is a thin
        pointer to `self.redimnet`, so the underlying `state_dict` is unaffected
        (weights stay under `redimnet.*`).
        """
        return self.redimnet

    @property
    def fbank_only(self) -> bool:
        """Whether to only extract (log-mel) features"""
        return getattr(self, "_fbank_only", False)

    @fbank_only.setter
    def fbank_only(self, value: bool):
        if hasattr(self, "receptive_field"):
            del self.receptive_field

        # `num_frames` is cached and depends on `fbank_only`
        self.num_frames.cache_clear()

        self._fbank_only = value

    def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract (log-mel) features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples) torch.Tensor

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins) torch.Tensor
            (log-mel) features, using the same layout as the WeSpeaker models.

        Note
        ----
        These are exactly the features consumed internally by `forward` and
        `forward_frames`.
        """
        return self.redimnet.compute_fbank(waveforms)

    @property
    def dimension(self) -> int:
        """Dimension of output"""

        if self.fbank_only:
            return self._num_mel_bins

        return self.redimnet.embed_dim

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

        # torchaudio's MelSpectrogram is centered (it pads the signal by `n_fft // 2`
        # on each side), so the number of (mel) frames is exactly `1 + samples // hop`.
        num_frames = 1 + num_samples // self._hop_length

        if self.fbank_only:
            return num_frames

        # the 1D pathway pools then up-samples the time axis back to the (mel) frame
        # rate, but the input is first truncated to a multiple of `time_stride`.
        time_stride = self.redimnet.backbone.time_stride
        return (num_frames // time_stride) * time_stride

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
        return conv1d_receptive_field_size(
            num_frames=num_frames,
            kernel_size=self._win_length,
            stride=self._hop_length,
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
        return conv1d_receptive_field_center(
            frame=frame,
            kernel_size=self._win_length,
            stride=self._hop_length,
            padding=0,
            dilation=1,
        )

    def forward_frames(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract frame-wise embeddings

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)

        Returns
        -------
        embeddings : (batch, dimension, embedding_frames) torch.Tensor
            Batch of frame-wise embeddings.
        """
        fbank = self.compute_fbank(waveforms)
        return self.redimnet.forward_frames(fbank)

    def forward_embedding(
        self, frames: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract speaker embeddings from frame-wise embeddings

        Parameters
        ----------
        frames : torch.Tensor
            Batch of frames with shape (batch, dimension, embedding_frames).
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights passed to the pooling layer.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings.
        """
        return self.redimnet.forward_embedding(frames, weights=weights)

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract speaker embeddings

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Batch of weights giving more weight to some frames than others (e.g. the
            activation of the speaker of interest). They are passed to the pooling
            layer, possibly after being interpolated to match the number of frames.
            When a `speakers` dimension is provided, one embedding is computed per
            speaker.

        Returns
        -------
        embeddings : (batch, dimension) or (batch, speakers, dimension) torch.Tensor
            Batch of embeddings. When `fbank_only` is set, the (log-mel) features
            with shape (batch, num_frames, num_mel_bins) are returned instead.
        """
        if self.fbank_only:
            return self.compute_fbank(waveforms)

        return self.redimnet(waveforms, weights=weights)


class ReDimNet2(BaseReDimNet2):
    """ReDimNet2 speaker embedding model

    Instantiates a ReDimNet2 model from a configuration dictionary. To load one of
    the official pretrained variants (`b0` ... `b6`), pass the `model_config`
    dictionary shipped alongside its weights, e.g.:

    >>> ckpt = torch.hub.load_state_dict_from_url(url)
    >>> model = ReDimNet2(config=ckpt["model_config"])
    >>> model.redimnet.load_state_dict(ckpt["state_dict"])

    When `config` is not provided, the default `ReDimNet2Wrap` architecture is used.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        # `ReDimNet2Wrap` expects `causal` to be one of {"none", "full", "only_1d"}
        # but defaults to the (invalid) boolean `False`, so make sure it is set.
        config = dict(config or {})
        config.setdefault("causal", "none")
        super().__init__(
            config=config,
            sample_rate=sample_rate,
            num_channels=num_channels,
            task=task,
        )


__all__ = ["BaseReDimNet2", "ReDimNet2"]
