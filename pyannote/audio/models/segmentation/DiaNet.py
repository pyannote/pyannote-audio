# MIT License
#
# Copyright (c) 2020 CNRS
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


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.core.utils.generators import pairwise


class DiaNet(Model):
    """DiaNet diarization model

    SincNet > Transformer Encoder > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    transformer : dict, optional
        Keyword arguments passed to the Transformer encoder.
        Defaults to {"d_model": 128, "dim_feedforward": 128, "num_layers": 4, "num_heads": 4}.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.

    """

    SINCNET_DEFAULTS = {"stride": 10}
    TRANSFORMER_DEFAULTS = {
        "d_model": 128,
        "dim_feedforward": 128,
        "num_layers": 4,
        "num_heads": 4,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sincnet: dict = None,
        transformer: dict = None,
        linear: dict = None,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet_hparams = dict(**self.SINCNET_DEFAULTS)
        if sincnet is not None:
            sincnet_hparams.update(**sincnet)
        sincnet_hparams["sample_rate"] = sample_rate
        self.hparams.sincnet = sincnet_hparams
        self.sincnet = SincNet(**self.hparams.sincnet)

        transformer_hparams = dict(**self.TRANSFORMER_DEFAULTS)
        if transformer is not None:
            transformer_hparams.update(**transformer)
        self.hparams.transformer = transformer_hparams

        self.to_transformer = nn.Linear(60, self.hparams.transformer["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.transformer["d_model"],
            dim_feedforward=self.hparams.transformer["dim_feedforward"],
            nhead=self.hparams.transformer["num_heads"],
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.hparams.transformer["num_layers"]
        )

        linear_hparams = dict(**self.LINEAR_DEFAULTS)
        if linear is not None:
            linear_hparams.update(**linear)
        self.hparams.linear = linear_hparams
        if self.hparams.linear["num_layers"] > 0:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [
                            self.hparams.transformer["d_model"],
                        ]
                        + [self.hparams.linear["hidden_size"]]
                        * self.hparams.linear["num_layers"]
                    )
                ]
            )

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.transformer["d_model"]

        self.classifier = nn.Linear(
            in_features, len(self.hparams.task_specifications.classes)
        )
        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)

        outputs = F.leaky_relu(
            self.to_transformer(
                rearrange(outputs, "batch feature frame -> frame batch feature")
            )
        )

        outputs = rearrange(
            self.transformer_encoder(outputs),
            "frame batch feature -> batch frame feature",
        )

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
