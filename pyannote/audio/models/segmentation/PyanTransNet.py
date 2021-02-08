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

import ipdb


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
        Defaults to {"nhead": 8, "num_encoder_layers": 6, "num_decoder_layers": 0, "dim_feedforward: 2048", "dropout: 0.1"},
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 1}
    TRANSFORMER_DEFAULT = {
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": 'relu',
    }

    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        transformer: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        ipdb.set_trace()

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        transformer = merge_dict(self.TRANSFORMER_DEFAULT, transformer)
        transformer["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("sincnet", "transformer", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)

        num_transformer_encoder_layers = transformer["num_encoder_layers"]
        del transformer["num_encoder_layers"]

        transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=60,  ** transformer)
        self.transformer = nn.TransformerEncoder(
            transformerEncoderLayer, num_transformer_encoder_layers)

        if linear["num_layers"] < 1:
            return

        transformer_out_features = 60

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        transformer_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

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
        ipdb.set_trace()
        outputs = self.sincnet(waveforms)

        outputs = self.transformer(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
