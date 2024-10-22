# MIT License
#
# Copyright 2024 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
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

# Initially copied from https://github.com/BUTSpeechFIT/DiariZen/blob/e747106e753bb17799602b24d396f60b13da81b4/diarizen/models/eend/model_wavlm_conformer.py


import contextlib
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.conformer import ConformerEncoder
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)


class DiariZen(Model):
    """Architecture used in Leveraging Self-Supervised Learning for Speaker Diarization

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_frozen: bool, optional
        Whether to freeze wav2vec weights. Defaults to False.
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    conformer : dict, optional
        Keyword arguments passed to the Conformer layer.

    Reference
    ---------
    Jiangyu Han, Federico Landini, Johan Rohdin, Anna Silnova, Mireia Diez, and Lukas Burget
    "Leveraging Self-Supervised Learning for Speaker Diarization"
    https://arxiv.org/abs/2409.09408
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    CONFORMER_DEFAULTS = {
        "attention_in": 256,
        "ffn_hidden": 1024,
        "num_head": 4,
        "num_layer": 4,
        "kernel_size": 31,
        "dropout": 0.1,
        "use_posi": False,
        "output_activate_function": False,
    }

    def __init__(
        self,
        wav2vec: Union[dict, str] = None,
        wav2vec_frozen: bool = False,
        wav2vec_layer: int = -1,
        conformer: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        wav2vec_dim: int
        wav2vec_num_layers: int

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(
                        f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz."
                    )
                wav2vec_dim = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()

            # `wav2vec` is a path to a self-supervised representation checkpoint
            else:
                _checkpoint = torch.load(wav2vec)
                wav2vec = _checkpoint.pop("config")
                self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                state_dict = _checkpoint.pop("state_dict")
                self.wav2vec.load_state_dict(state_dict)
                wav2vec_dim = wav2vec["encoder_embed_dim"]
                wav2vec_num_layers = wav2vec["encoder_num_layers"]

        # `wav2vec` is a config dictionary understood by `wav2vec2_model`
        # this branch is typically used by Model.from_pretrained(...)
        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )

        conformer = merge_dict(self.CONFORMER_DEFAULTS, conformer)

        self.save_hyperparameters(
            "wav2vec", "wav2vec_frozen", "wav2vec_layer", "conformer"
        )

        self.conformer = ConformerEncoder(**conformer)
        self.proj = nn.Linear(wav2vec_dim, conformer["attention_in"])
        self.lnorm = nn.LayerNorm(conformer["attention_in"])

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("DiariZen does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        self.classifier = nn.Linear(
            self.hparams.conformer["attention_in"], self.dimension
        )
        self.activation = self.default_activation()

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

        num_frames = num_samples
        for conv_layer in self.wav2vec.feature_extractor.conv_layers:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )

        # TODO: look at conformer.num_frames

        return num_frames

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

        # TODO: look at conformer receptive field size

        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

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

        # TODO: look at conformer receptive field center

        receptive_field_center = frame
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        num_layers = (
            None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        )

        context = (
            torch.no_grad() if self.hparams.wav2vec_frozen else contextlib.nullcontext()
        )
        with context:
            outputs, _ = self.wav2vec.extract_features(
                waveforms.squeeze(1), num_layers=num_layers
            )

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.wav2vec_weights, dim=0
            )
        else:
            outputs = outputs[-1]

        outputs = self.proj(outputs)
        outputs = self.lnorm(outputs)
        outputs = self.conformer(outputs)
        return self.activation(self.classifier(outputs))
