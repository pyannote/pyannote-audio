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

from typing import Literal, Optional
from warnings import warn
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import ArcFaceLoss

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.powerset import Powerset
from pyannote.core.utils.generators import pairwise


# TODO deplace these two lines into uitls/multi_task
Subtask = Literal["diarization", "embedding"]
Subtasks = list(Subtask.__args__)


class SpeakerEndToEndDiarization(Model):
    """Speaker End-to-End Diarization and Embedding model
    SINCNET -- TDNN .. TDNN -- TDNN ..TDNN -- StatsPool -- Linear --  Classifier
                                    \ LSTM ... LSTM -- FeedForward -- Classifier
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
            lstm: dict= None,
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
            warn("For now, the model only support a number of features of 60. Set it to 60")
            num_features = 60
        self.num_features = num_features
        self.separation_idx = separation_idx
        self.save_hyperparameters("num_features", "embedding_dim", "separation_idx")


        # sincnet module
        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        self.sincnet =SincNet(**sincnet)
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

        self.arc_face_loss = ArcFaceLoss(
            len(self.specifications[Subtasks.index("embedding")].classes),
            self.hparams["embedding_dim"],
            margin=self.task.margin,
            scale=self.task.scale,
        )


    def forward(self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            diarization_outputs = rearrange(common_outputs, "batch feature frame -> batch frame feature")
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
        embedding_outputs = torch.clone(common_outputs)
        for tdnn_block in self.tdnn_blocks[tdnn_idx:]:
            embedding_outputs = tdnn_block(embedding_outputs)
        embedding_outputs = self.stats_pool(embedding_outputs, weights=weights)
        embedding_outputs = self.embedding(embedding_outputs)

        return (diarization_outputs, embedding_outputs)
