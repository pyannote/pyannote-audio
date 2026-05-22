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

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)


class InteractiveSSeRiouSS(Model):
    """Self-Supervised Representation for Interactive Speaker Segmentation

    Extends SSeRiouSS with an optional interaction tensor injected at the LSTM input.
    The interaction tensor has shape (batch, frames, interaction_size) and encodes
    user-provided hints (e.g. overlap region, speaker change point, number of speakers).

    At inference time, build the interaction tensor as all-zeros for no hints, or
    fill specific channels according to `model.hparams.interaction_slices` to inject
    hints. The model's forward pass accepts interactions as an optional argument:

        output = model(waveform)                        # no hints
        output = model(waveform, interactions=tensor)   # with hints

    Parameters
    ----------
    wav2vec : dict or str, optional
        Wav2vec backbone. Either a torchaudio pipeline name (e.g. "WAVLM_BASE"),
        a path to a checkpoint, or a dict of wav2vec2_model kwargs.
        Defaults to "WAVLM_BASE".
    wav2vec_frozen : bool, optional
        Whether to freeze wav2vec weights. Defaults to False.
    wav2vec_layer : int, optional
        Index of wav2vec layer to use as LSTM input.
        Defaults to -1 (weighted average of all layers).
    lstm : dict, optional
        Keyword arguments for the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 4, "bidirectional": True}.
    linear : dict, optional
        Keyword arguments for the linear layers.
        Defaults to {"hidden_size": 128, "num_layers": 2}.
    sample_rate : int, optional
        Audio sample rate. Defaults to 16000.
    num_channels : int, optional
        Number of audio channels. Defaults to 1 (mono).
    task : Task, optional
        Training task. Provides interaction_size and interaction_slices when
        given. Not needed for inference from a saved checkpoint.
    interaction_size : int, optional
        Total number of interaction channels. Inferred from task if not given.
        Saved as hyperparameter so it is available when loading from checkpoint.
    interaction_slices : dict[str, tuple[int, int]], optional
        Mapping from interaction name to (start_channel, end_channel) indices.
        Inferred from task if not given. Saved as hyperparameter so the channel
        layout is self-contained in the checkpoint and available at inference time
        without needing the original task.
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        wav2vec: Optional[dict | str] = None,
        wav2vec_frozen: bool = False,
        wav2vec_layer: int = -1,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        interaction_size: Optional[int] = None,
        interaction_slices: Optional[dict] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        interaction_size = interaction_size or getattr(task, "interaction_size", None)
        interaction_slices = interaction_slices or getattr(task, "_interaction_slices", None)

        if isinstance(wav2vec, str):
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(
                        f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz."
                    )
                wav2vec_dim = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()

            else:
                _checkpoint = torch.load(wav2vec, map_location="cpu")

                if "config" in _checkpoint:
                    wav2vec = _checkpoint["config"]
                    self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                    state_dict = _checkpoint["state_dict"]
                    self.wav2vec.load_state_dict(state_dict)
                    wav2vec_dim = wav2vec["encoder_embed_dim"]
                    wav2vec_num_layers = wav2vec["encoder_num_layers"]

                elif "hyper_parameters" in _checkpoint and isinstance(
                    _checkpoint["hyper_parameters"].get("wav2vec"), dict
                ):
                    wav2vec = _checkpoint["hyper_parameters"]["wav2vec"]
                    self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                    wav2vec_dim = wav2vec["encoder_embed_dim"]
                    wav2vec_num_layers = wav2vec["encoder_num_layers"]

                else:
                    raise ValueError(
                        "Unsupported checkpoint format for wav2vec. Expected keys "
                        "'config' (SSL) or 'hyper_parameters[\"wav2vec\"]' (Lightning)."
                    )

        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )

        for param in self.wav2vec.parameters():
            param.requires_grad = not wav2vec_frozen

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters(
            "wav2vec",
            "wav2vec_frozen",
            "wav2vec_layer",
            "lstm",
            "linear",
            "interaction_size",
            "interaction_slices",
        )

        self.lstm = nn.LSTM(wav2vec_dim + self.hparams.interaction_size, **lstm)

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [lstm_out_features]
                    + [self.hparams.linear["hidden_size"]] * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        if isinstance(self.specifications, tuple):
            raise ValueError("InteractiveSSeRiouSS does not support multi-tasking.")
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )
        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    def _wav2vec_conv_layers(self):
        """Return conv_layers regardless of torchaudio version wrapping."""
        wav2vec = getattr(self.wav2vec, "model", self.wav2vec)
        return wav2vec.feature_extractor.conv_layers

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        num_frames = num_samples
        for conv_layer in self._wav2vec_conv_layers():
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        receptive_field_size = num_frames
        for conv_layer in reversed(self._wav2vec_conv_layers()):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        receptive_field_center = frame
        for conv_layer in reversed(self._wav2vec_conv_layers()):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def forward(
        self,
        waveforms: torch.Tensor,
        interactions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        interactions : (batch, frame, interaction_size), optional
            Interaction tensor. Defaults to all-zeros (no hints).

        Returns
        -------
        scores : (batch, frame, classes)
        """
        num_layers = None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        outputs, _ = self.wav2vec.extract_features(
            waveforms.squeeze(1), num_layers=num_layers
        )

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.wav2vec_weights, dim=0)
        else:
            outputs = outputs[-1]

        if interactions is None:
            batch_size, num_frames, _ = outputs.shape
            interactions = torch.zeros(
                batch_size,
                num_frames,
                self.hparams.interaction_size,
                device=outputs.device,
                dtype=outputs.dtype,
            )

        outputs = F.normalize(outputs, p=2, dim=2)
        outputs = torch.cat([outputs, interactions], dim=2)
        outputs, _ = self.lstm(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
