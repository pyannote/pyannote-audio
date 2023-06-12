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
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from asteroid.masknn.convolutional import TDConvNet
from asteroid_filterbanks import make_enc_dec
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid.masknn import DPRNN


class SepDiarNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    ENCODER_DECODER_DEFAULTS = {
        "fb_name": "stft",
        "kernel_size": 512,
        "n_filters": 512,
        "stride": 256,
    }
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    CONVNET_DEFAULTS = {
        "n_blocks": 8,
        "n_repeats": 3,
        "bn_chan": 128,
        "hid_chan": 512,
        "skip_chan": 128,
        "conv_kernel_size": 3,
        "norm_type": "gLN",
        "mask_act": "relu",
    }
    DPRNN_DEFAULTS = {
        "n_repeats": 6,
        "bn_chan": 128,
        "hid_size": 128,
        "chunk_size": 100,
        "norm_type": "gLN",
        "mask_act": "relu",
        "rnn_type": "LSTM",
    }

    def __init__(
        self,
        encoder_decoder: dict = None,
        lstm: dict = None,
        linear: dict = None,
        convnet: dict = None,
        dprnn: dict = None,
        free_encoder: dict = None,
        stft_encoder: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        encoder_type: str = None,
        n_sources: int = 3,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        convnet = merge_dict(self.CONVNET_DEFAULTS, convnet)
        dprnn = merge_dict(self.DPRNN_DEFAULTS, dprnn)
        encoder_decoder = merge_dict(self.ENCODER_DECODER_DEFAULTS, encoder_decoder)
        self.save_hyperparameters("encoder_decoder", "lstm", "linear", "convnet", "dprnn")

        if encoder_decoder["fb_name"] == "free":
            n_feats_out = encoder_decoder["n_filters"]
        elif encoder_decoder["fb_name"] == "stft":
            n_feats_out = int(2 * (encoder_decoder["n_filters"] / 2 + 1))
        else:
            raise ValueError("Filterbank type not recognized.")
        self.encoder, self.decoder = make_enc_dec(
            sample_rate=sample_rate, **self.hparams.encoder_decoder
        )
        self.masker = DPRNN(n_feats_out, n_src=n_sources, **self.hparams.dprnn)
        #self.convnet= TDConvNet(n_feats_out, **self.hparams.convnet)

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(n_sources * n_feats_out, **multi_layer_lstm)

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
                        6 * n_feats_out
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm
                    )
                    for i in range(num_layers)
                ]
            )

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

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        # if isinstance(self.specifications, tuple):
        #     raise ValueError("PyanNet does not support multi-tasking.")

        # if self.specifications.powerset:
        out_features = self.specifications[0].num_powerset_classes
        # else:
        #     out_features = len(self.specifications.classes)

        self.classifier = nn.Linear(in_features, out_features)
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

        tf_rep = self.encoder(waveforms)
        masks = self.masker(tf_rep)

        masked_tf_rep = masks * tf_rep.unsqueeze(1)
        decoded_sources = self.decoder(masked_tf_rep)
        decoded_sources = pad_x_to_y(decoded_sources, waveforms)
        decoded_sources = decoded_sources.transpose(1, 2)

        outputs = rearrange(
            masks, "batch nsrc nfilters nframes -> batch nframes nfilters nsrc"
        )
        outputs = torch.flatten(outputs, start_dim=2, end_dim=3)

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation[0](self.classifier(outputs)), decoded_sources
