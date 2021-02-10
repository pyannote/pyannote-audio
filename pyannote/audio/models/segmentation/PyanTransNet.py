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
from pyannote.audio.utils.params import merge_dict
from pyannote.core.utils.generators import pairwise


class PyanTransNet(Model):
    """PyanTransNet segmentation model

    SincNet > Transformer -> Transformer output
                   |                             -> Feed forward -> Classifer
                    -> context vector


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
        Keyword arguments passed to the Transformer layer.
        Defaults to {"nhead": 4, "num_encoder_layers": 1,  "dim_feedforward: 2048", "dropout: 0.1", "afterlstm": True},
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 1}
    TRANSFORMER_DEFAULT = {
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": 'relu',
        "afterlstm": True,
    }

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }

    LINEAR_DEFAULTS = {"hidden_size": 60, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        lstm: dict = None,
        transformer: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True

        transformer = merge_dict(self.TRANSFORMER_DEFAULT, transformer)
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "transformer", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)

        num_transformer_encoder_layers = transformer["num_encoder_layers"]
        del transformer["num_encoder_layers"]

        self.afterlstm = transformer["afterlstm"]
        del transformer["afterlstm"]

        transformer_input_dim = 60

        if lstm["num_layers"] > 0:
            if self.afterlstm:
                transformer_input_dim = lstm["hidden_size"]
            monolithic = lstm["monolithic"]
            if monolithic:
                multi_layer_lstm = dict(lstm)
                del multi_layer_lstm["monolithic"]
                self.lstm = nn.LSTM(60, **multi_layer_lstm)

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
                            60
                            if i == 0
                            else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                            **one_layer_lstm
                        )
                        for i in range(num_layers)
                    ]
                )

        transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,  ** transformer)
        self.transformer = nn.TransformerEncoder(
            transformerEncoderLayer, num_transformer_encoder_layers)

        if linear["num_layers"] < 1:
            return

        transformer_output_dim = transformer_input_dim

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        transformer_output_dim,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.lstm_numlayers = lstm["num_layers"]

    def build(self):
        in_features = 60
        self.classifier = nn.Linear(
            in_features, len(self.specifications.classes))
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

        if self.lstm_numlayers > 0:
            if self.afterlstm:
                if self.hparams.lstm["monolithic"]:
                    outputs, _ = self.lstm(
                        rearrange(
                            outputs, "batch feature frame -> batch frame feature")
                    )
                else:
                    outputs = rearrange(
                        outputs, "batch feature frame -> batch frame feature")
                    for i, lstm in enumerate(self.lstm):
                        outputs, _ = lstm(outputs)
                        if i + 1 < self.hparams.lstm["num_layers"]:
                            outputs = self.dropout(outputs)
                    outputs = self.transformer(outputs)
            else:
                outputs = rearrange(
                    outputs, "batch feature frame -> batch frame feature")
                outputs = self.transformer(outputs)
                if self.hparams.lstm["monolithic"]:
                    outputs, _ = self.lstm(
                        outputs
                    )
                else:
                    for i, lstm in enumerate(self.lstm):
                        outputs, _ = lstm(outputs)
                        if i + 1 < self.hparams.lstm["num_layers"]:
                            outputs = self.dropout(outputs)
        else:
            outputs = rearrange(
                outputs, "batch feature frame -> batch frame feature")
            outputs = self.transformer(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
