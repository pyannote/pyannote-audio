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

import os
from functools import lru_cache
from typing import Literal, Optional, Union
from warnings import warn

import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise
from torch import nn

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.powerset import Powerset

# TODO deplace these two lines into uitls/multi_task
Subtask = Literal["diarization", "embedding"]
Subtasks = list(Subtask.__args__)


class WeSpeakerBasesEndToEndDiarization(Model):
    """
    WeSpeaker-based joint speaker diarization and speaker
    embedding extraction model
    """

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate=16000,
        embedding_dim=256,
        num_channels=1,
        task: Optional[Union[Task, None]] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        # speakers embedding extraction submodel:
        self.resnet34 = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self.embedding_dim = embedding_dim
        self.save_hyperparameters("embedding_dim")

        # speaker segmentation submodel:
        self.pyannet = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=os.environ["HG_TOKEN"],
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        """"""
        dia_specs = self.specifications[Subtasks.index("diarization")]
        self.powerset = Powerset(
            len(dia_specs.classes),
            dia_specs.powerset_max_classes,
        )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.pyannet.sincnet.num_frames(num_samples)

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
        return self.pyannet.sincnet.receptive_field_size(num_frames=num_frames)

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

        return self.pyannet.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveformms: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        """
        dia_outputs = self.pyannet(waveformms)
        weights = self.powerset.to_multilabel(dia_outputs, soft=True)
        weights = rearrange(weights, "b f s -> b s f")
        emb_outputs = self.resnet34(waveformms, weights)
        return (dia_outputs, emb_outputs)


class SpeakerEndToEndDiarization(Model):
    """Speaker End-to-End Diarization and Embedding model
    SINCNET -- TDNN .. TDNN -- TDNN ..TDNN -- StatsPool -- Linear --  Classifier
                                    \\ LSTM ... LSTM -- FeedForward -- Classifier
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
        "batch_first": True,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_features: int = 60,
        embedding_dim: int = 512,
        separation_idx: int = 2,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if num_features != 60:
            warn(
                "For now, the model only support a number of features of 60. Set it to 60"
            )
            num_features = 60
        self.num_features = num_features
        self.separation_idx = separation_idx
        self.save_hyperparameters("num_features", "embedding_dim", "separation_idx")

        # sincnet module
        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        self.sincnet = SincNet(**sincnet)
        self.save_hyperparameters("sincnet")

        # tdnn modules
        self.tdnn_blocks = nn.ModuleList()
        in_channel = num_features
        out_channels = [512, 512, 512, 512, 1500]
        kernel_sizes = [5, 3, 3, 1, 1]
        dilations = [1, 2, 3, 1, 1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, kernel_sizes, dilations
        ):
            self.tdnn_blocks.extend(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding="same",
                        ),
                        nn.LeakyReLU(),
                        nn.BatchNorm1d(out_channel),
                    ),
                ]
            )
            in_channel = out_channel

        # lstm modules:
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        self.save_hyperparameters("lstm")
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(out_channels[separation_idx], **multi_layer_lstm)
        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            del one_layer_lstm["monolithic"]
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        out_channels[separation_idx]
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        # linear module for the diarization part:
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("linear")
        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
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

        # stats pooling module for the embedding part:
        self.stats_pool = StatsPool()
        # linear module for the embedding part:
        self.embedding = nn.Linear(in_channel * 2, embedding_dim)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        diarization_spec = self.specifications[Subtasks.index("diarization")]
        out_features = diarization_spec.num_powerset_classes
        self.classifier = nn.Linear(in_features, out_features)
        self.powerset = Powerset(
            len(diarization_spec.classes),
            diarization_spec.powerset_max_classes,
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
            Batch of weights wiht shape (batch, frame)
        """
        common_outputs = self.sincnet(waveforms)
        # (batch, features, frames)
        # common part to diarization and embedding:
        tdnn_idx = 0
        while tdnn_idx <= self.separation_idx:
            common_outputs = self.tdnn_blocks[tdnn_idx](common_outputs)
            tdnn_idx = tdnn_idx + 1
        # diarization part:
        if self.hparams.lstm["monolithic"]:
            diarization_outputs, _ = self.lstm(
                rearrange(common_outputs, "batch feature frame -> batch frame feature")
            )
        else:
            diarization_outputs = rearrange(
                common_outputs, "batch feature frame -> batch frame feature"
            )
            for i, lstm in enumerate(self.lstm):
                diarization_outputs, _ = lstm(diarization_outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    diarization_outputs = self.linear()

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                diarization_outputs = F.leaky_relu(linear(diarization_outputs))
        diarization_outputs = self.classifier(diarization_outputs)
        diarization_outputs = F.log_softmax(diarization_outputs, dim=-1)
        weights = self.powerset(diarization_outputs).transpose(1, 2)

        # embedding part:
        embedding_outputs = common_outputs
        for tdnn_block in self.tdnn_blocks[tdnn_idx:]:
            embedding_outputs = tdnn_block(embedding_outputs)
        embedding_outputs = self.stats_pool(embedding_outputs, weights=weights)
        embedding_outputs = self.embedding(embedding_outputs)

        return (diarization_outputs, embedding_outputs)


class SpeakerEndToEndDiarizationV2(Model):
    """This version uses a LSTM encoder in the embedding branch instead StatsPool block"""

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
        "batch_first": True,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_features: int = 60,
        embedding_dim: int = 512,
        separation_idx: int = 2,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if num_features != 60:
            warn(
                "For now, the model only support a number of features of 60. Set it to 60"
            )
            num_features = 60
        self.num_features = num_features
        self.separation_idx = separation_idx
        self.save_hyperparameters("num_features", "embedding_dim", "separation_idx")

        # sincnet module
        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        self.sincnet = SincNet(**sincnet)
        self.save_hyperparameters("sincnet")

        # tdnn modules
        self.tdnn_blocks = nn.ModuleList()
        in_channel = num_features
        out_channels = [512, 512, 512, 512, 1500]
        kernel_sizes = [5, 3, 3, 1, 1]
        dilations = [1, 2, 3, 1, 1]
        self.last_tdnn_out_channels = out_channels[-1]

        for out_channel, kernel_size, dilation in zip(
            out_channels, kernel_sizes, dilations
        ):
            self.tdnn_blocks.extend(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding="same",
                        ),
                        nn.LeakyReLU(),
                        nn.BatchNorm1d(out_channel),
                    ),
                ]
            )
            in_channel = out_channel

        # lstm modules:
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        self.save_hyperparameters("lstm")
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(out_channels[separation_idx], **multi_layer_lstm)
        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            del one_layer_lstm["monolithic"]
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        out_channels[separation_idx]
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

        # linear module for the diarization part:
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("linear")
        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
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

        # linear module for the embedding part:
        self.embedding = nn.Linear(2 * self.last_tdnn_out_channels, embedding_dim)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        diarization_spec = self.specifications[Subtasks.index("diarization")]
        out_features = diarization_spec.num_powerset_classes
        self.classifier = nn.Linear(in_features, out_features)

        self.powerset = Powerset(
            len(diarization_spec.classes),
            diarization_spec.powerset_max_classes,
        )

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.encoder = nn.LSTM(
            # number of channel in the outputs of the last TDNN layer + lstm_out_features
            input_size=self.last_tdnn_out_channels + lstm_out_features,
            hidden_size=len(diarization_spec.classes) * self.last_tdnn_out_channels,
            batch_first=True,
            bidirectional=True,
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
            Batch of weights wiht shape (batch, frame)
        """
        common_outputs = self.sincnet(waveforms)
        # (batch, features, frames)
        # common part to diarization and embedding:
        tdnn_idx = 0
        while tdnn_idx <= self.separation_idx:
            common_outputs = self.tdnn_blocks[tdnn_idx](common_outputs)
            tdnn_idx = tdnn_idx + 1
        # diarization part:
        dia_outputs = common_outputs

        if self.hparams.lstm["monolithic"]:
            dia_outputs, _ = self.lstm(
                rearrange(dia_outputs, "batch feature frame -> batch frame feature")
            )
        else:
            dia_outputs = rearrange(
                common_outputs, "batch feature frame -> batch frame feature"
            )
            for i, lstm in enumerate(self.lstm):
                dia_outputs, _ = lstm(dia_outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    dia_outputs = self.linear()
        lstm_outputs = dia_outputs
        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                dia_outputs = F.leaky_relu(linear(dia_outputs))
        dia_outputs = self.classifier(dia_outputs)
        dia_outputs = F.log_softmax(dia_outputs, dim=-1)

        # embedding part:
        emb_outputs = common_outputs
        for tdnn_block in self.tdnn_blocks[tdnn_idx:]:
            emb_outputs = tdnn_block(emb_outputs)

        emb_outputs = rearrange(emb_outputs, "b c f -> b f c")
        # Concatenation of last tdnn layer outputs with the last diarization lstm outputs:
        emb_outputs = torch.cat((emb_outputs, lstm_outputs), dim=2)
        _, emb_outputs = self.encoder(emb_outputs)
        emb_outputs = rearrange(emb_outputs[0], "l b h -> b (l h)")
        emb_outputs = torch.reshape(
            emb_outputs, (emb_outputs.shape[0], self.powerset.num_classes, -1)
        )
        emb_outputs = self.embedding(emb_outputs)

        return (dia_outputs, emb_outputs)
