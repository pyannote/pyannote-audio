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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise
from torch import nn

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.powerset import Powerset

from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

import torchaudio

# TODO deplace these two lines into uitls/multi_task
Subtask = Literal["diarization", "embedding"]
Subtasks = list(Subtask.__args__)


class WavLMEnd2EndDiarization(Model):
    """Self-Supervised representation for joint speaker diarization
    and speaker embeddings extraction


    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    freeze_wav2vec: bool, optional
        Whether to freeze wa2vec. Default to true
    emb_dim: int, optional
        Dimension of the speaker embedding in output
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }

    LINEAR_DEFAULT = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        freeze_wav2vec: bool = True,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        embedding_dim: Optional[int] = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle.sample_rate:
                    raise ValueError(
                        f"Expected {bundle.sample_rate}Hz, found {sample_rate}Hz."
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
            # weighting parameters for the diarization branch
            self.dia_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
            # weighting parameters for the embedding branch
            self.emb_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
        self.save_hyperparameters("wav2vec", "wav2vec_layer", "freeze_wav2vec")

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULT, linear)
        self.save_hyperparameters("lstm", "linear")
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(wav2vec_dim, **multi_layer_lstm)
        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        (
                            wav2vec_dim
                            if i == 0
                            else lstm["hidden_size"]
                            * (2 if lstm["bidirectional"] else 1)
                        ),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return
        lstm_out_features = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.pooling = StatsPool(computde_std=False)
        self.embeddings = nn.Linear(wav2vec_dim, embedding_dim)

        self.save_hyperparameters("embedding_dim")

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.specifications[Subtasks.index("diarization")].num_powerset_classes

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

        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
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

    def build(self):
        """"""
        max_num_speaker_per_chunk = len(self.specifications[Subtasks.index("diarization")].classes)
        max_num_speaker_per_frame = self.specifications[Subtasks.index("diarization")].powerset_max_classes
        self.powerset = Powerset(
            max_num_speaker_per_chunk,
            max_num_speaker_per_frame
        )

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            lstm = self.hparams.lstm
            in_features = lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1)

        self.classifier = nn.Linear(in_features, self.dimension)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        diarization, embeddings : (batch, frames, classes), (batch, num_speaker, embed_dim)
        """

        num_layers = (
            None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        )

        if self.hparams.freeze_wav2vec:
            with torch.no_grad():
                outputs, _ = self.wav2vec.extract_features(
                    waveforms.squeeze(1), num_layers=num_layers
                )
        else:
            outputs, _ = self.wav2vec.extract_features(
                waveforms.squeeze(1), num_layers=num_layers
            )

        if num_layers is None:
            dia_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.dia_wav2vec_weights, dim=0
            )
            emb_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.emb_wav2vec_weights, dim=0
            )
        else:
            dia_outputs = emb_outputs = outputs[-1]

        if self.hparams.lstm["monolithic"]:
            dia_outputs, _ = self.lstm(dia_outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                dia_outputs, _ = lstm(dia_outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    dia_outputs = self.dropout(dia_outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                dia_outputs = F.leaky_relu(linear(dia_outputs))
        dia_outputs = self.classifier(dia_outputs)
        dia_outputs = F.log_softmax(dia_outputs, dim=-1)

        weights = self.powerset.to_multilabel(dia_outputs, soft=True)
        weights = rearrange(weights, "b f s -> b s f")
        emb_outputs = rearrange(emb_outputs, "b f w -> b w f")
        emb_outputs = self.pooling(emb_outputs, weights)
        emb_outputs = self.embeddings(emb_outputs)

        return (dia_outputs, emb_outputs)


class WavLMEnd2EndDiarizationv2(Model):
    """Self-Supervised representation for joint speaker diarization
    and speaker embeddings extraction


    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    freeze_wav2vec: bool, optional
        Whether to freeze wa2vec. Default to true
    emb_dim: int, optional
        Dimension of the speaker embedding in output
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }

    LINEAR_DEFAULT = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        freeze_wav2vec: bool = True,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        embedding_dim: Optional[int] = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle.sample_rate:
                    raise ValueError(
                        f"Expected {bundle.sample_rate}Hz, found {sample_rate}Hz."
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
            # weighting parameters for the diarization branch
            self.dia_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
            # weighting parameters for the embedding branch
            self.emb_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
        self.save_hyperparameters("wav2vec", "wav2vec_layer", "freeze_wav2vec")

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULT, linear)
        self.save_hyperparameters("lstm", "linear")
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(wav2vec_dim, **multi_layer_lstm)
        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        (
                            wav2vec_dim
                            if i == 0
                            else lstm["hidden_size"]
                            * (2 if lstm["bidirectional"] else 1)
                        ),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return
        lstm_out_features = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.pooling = StatsPool(computde_std=False)
        self.embeddings = nn.Sequential(
                nn.Linear(in_features=wav2vec_dim, out_features=1024),
                nn.LeakyReLU(),
                nn.Linear(in_features=1024, out_features=embedding_dim),
        )

        self.save_hyperparameters("embedding_dim")

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.specifications[Subtasks.index("diarization")].num_powerset_classes

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

        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
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

    def build(self):
        """"""
        max_num_speaker_per_chunk = len(self.specifications[Subtasks.index("diarization")].classes)
        max_num_speaker_per_frame = self.specifications[Subtasks.index("diarization")].powerset_max_classes
        self.powerset = Powerset(
            max_num_speaker_per_chunk,
            max_num_speaker_per_frame
        )

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            lstm = self.hparams.lstm
            in_features = lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1)

        self.classifier = nn.Linear(in_features, self.dimension)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        diarization, embeddings : (batch, frames, classes), (batch, num_speaker, embed_dim)
        """

        num_layers = (
            None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        )

        if self.hparams.freeze_wav2vec:
            with torch.no_grad():
                outputs, _ = self.wav2vec.extract_features(
                    waveforms.squeeze(1), num_layers=num_layers
                )
        else:
            outputs, _ = self.wav2vec.extract_features(
                waveforms.squeeze(1), num_layers=num_layers
            )

        if num_layers is None:
            dia_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.dia_wav2vec_weights, dim=0
            )
            emb_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.emb_wav2vec_weights, dim=0
            )
        else:
            dia_outputs = emb_outputs = outputs[-1]

        if self.hparams.lstm["monolithic"]:
            dia_outputs, _ = self.lstm(dia_outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                dia_outputs, _ = lstm(dia_outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    dia_outputs = self.dropout(dia_outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                dia_outputs = F.leaky_relu(linear(dia_outputs))
        dia_outputs = self.classifier(dia_outputs)
        dia_outputs = F.log_softmax(dia_outputs, dim=-1)

        weights = self.powerset.to_multilabel(dia_outputs, soft=True)
        weights = rearrange(weights, "b f s -> b s f")
        emb_outputs = rearrange(emb_outputs, "b f w -> b w f")
        emb_outputs = self.pooling(emb_outputs, weights)
        emb_outputs = self.embeddings(emb_outputs)

        return (dia_outputs, emb_outputs)


class WavLMEnd2EndDiarizationv3(Model):
    """With modified weights

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    freeze_wav2vec: bool, optional
        Whether to freeze wa2vec. Default to true
    emb_dim: int, optional
        Dimension of the speaker embedding in output
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }

    LINEAR_DEFAULT = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        freeze_wav2vec: bool = True,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        embedding_dim: Optional[int] = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle.sample_rate:
                    raise ValueError(
                        f"Expected {bundle.sample_rate}Hz, found {sample_rate}Hz."
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
            # weighting parameters for the diarization branch
            self.dia_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
            # weighting parameters for the embedding branch
            self.emb_wav2vec_weights = nn.Parameter(
                data=torch.ones(wav2vec_num_layers), requires_grad=True
            )
        self.save_hyperparameters("wav2vec", "wav2vec_layer", "freeze_wav2vec")

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULT, linear)
        self.save_hyperparameters("lstm", "linear")
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(wav2vec_dim, **multi_layer_lstm)
        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        (
                            wav2vec_dim
                            if i == 0
                            else lstm["hidden_size"]
                            * (2 if lstm["bidirectional"] else 1)
                        ),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return
        lstm_out_features = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.pooling = StatsPool(computde_std=False)
        self.embeddings = nn.Sequential(
                nn.Linear(in_features=wav2vec_dim, out_features=1024),
                nn.LeakyReLU(),
                nn.Linear(in_features=1024, out_features=embedding_dim),
        )

        self.save_hyperparameters("embedding_dim")

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.specifications[Subtasks.index("diarization")].num_powerset_classes

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

        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
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

    def build(self):
        """"""
        max_num_speaker_per_chunk = len(self.specifications[Subtasks.index("diarization")].classes)
        max_num_speaker_per_frame = self.specifications[Subtasks.index("diarization")].powerset_max_classes
        self.powerset = Powerset(
            max_num_speaker_per_chunk,
            max_num_speaker_per_frame
        )

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            lstm = self.hparams.lstm
            in_features = lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1)

        self.classifier = nn.Linear(in_features, self.dimension)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        diarization, embeddings : (batch, frames, classes), (batch, num_speaker, embed_dim)
        """

        num_layers = (
            None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        )

        if self.hparams.freeze_wav2vec:
            with torch.no_grad():
                outputs, _ = self.wav2vec.extract_features(
                    waveforms.squeeze(1), num_layers=num_layers
                )
        else:
            outputs, _ = self.wav2vec.extract_features(
                waveforms.squeeze(1), num_layers=num_layers
            )

        if num_layers is None:
            dia_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.dia_wav2vec_weights, dim=0
            )
            emb_outputs = torch.stack(outputs, dim=-1) @ F.softmax(
                self.emb_wav2vec_weights, dim=0
            )
        else:
            dia_outputs = emb_outputs = outputs[-1]

        if self.hparams.lstm["monolithic"]:
            dia_outputs, _ = self.lstm(dia_outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                dia_outputs, _ = lstm(dia_outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    dia_outputs = self.dropout(dia_outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                dia_outputs = F.leaky_relu(linear(dia_outputs))
        dia_outputs = self.classifier(dia_outputs)
        dia_outputs = F.log_softmax(dia_outputs, dim=-1)

        # hard-segmentation in multilabel space
        multilabel_segmentations: torch.Tensor = self.powerset.to_multilabel(dia_outputs)
        # (batch_size, num_frames, max_speakers_per_chunk), {0, 1}

        weights = (
            (
                F.one_hot(
                    torch.argmax(dia_outputs, dim=2),
                    num_classes=self.powerset.num_powerset_classes,
                )[:, :, 1 : 1 + self.powerset.num_classes]
                + 1e-2
            )
            * multilabel_segmentations
        ).transpose(2, 1)
        # (batch_size, max_speakers_per_chunk, num_frames)
        # 0.000 if speaker is inactive
        # 0.001 if speaker is active but not alone
        # 1.001 if speaker is active and alone

        emb_outputs = rearrange(emb_outputs, "b f w -> b w f")
        emb_outputs = self.pooling(emb_outputs, weights)
        emb_outputs = self.embeddings(emb_outputs)

        return (dia_outputs, emb_outputs)
