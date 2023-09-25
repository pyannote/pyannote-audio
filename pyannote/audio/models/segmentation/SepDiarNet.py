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

    ENCODER_DECODER_DEFAULTS = {
        "fb_name": "stft",
        "kernel_size": 512,
        "n_filters": 64,
        "stride": 32,
    }
    LSTM_DEFAULTS = {
        "hidden_size": 64,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 64, "num_layers": 0}
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
        use_lstm: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        convnet = merge_dict(self.CONVNET_DEFAULTS, convnet)
        dprnn = merge_dict(self.DPRNN_DEFAULTS, dprnn)
        encoder_decoder = merge_dict(self.ENCODER_DECODER_DEFAULTS, encoder_decoder)
        self.n_src = n_sources
        self.use_lstm = use_lstm
        self.save_hyperparameters(
            "encoder_decoder", "lstm", "linear", "convnet", "dprnn"
        )
        self.learning_rate = lr
        self.n_sources = n_sources

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

        # diarization can use a lower resolution than separation, use 128x downsampling
        diarization_scaling = int(128 / encoder_decoder["stride"])
        self.average_pool = nn.AvgPool1d(
            diarization_scaling, stride=diarization_scaling
        )
        lstm_out_features = n_feats_out
        if self.use_lstm:
            del lstm["monolithic"]
            multi_layer_lstm = dict(lstm)
            self.lstm = nn.LSTM(n_feats_out, **multi_layer_lstm)
            lstm_out_features = lstm["hidden_size"] * (
                2 if lstm["bidirectional"] else 1
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
        if self.use_lstm or self.hparams.linear["num_layers"] > 0:
            self.classifier = nn.Linear(64, 1)
        else:
            self.classifier = nn.Linear(1, 1)
        self.activation = self.default_activation()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        bsz = waveforms.shape[0]
        tf_rep = self.encoder(waveforms)
        masks = self.masker(tf_rep)
        # shape: (batch, nsrc, nfilters, nframes)
        masked_tf_rep = masks * tf_rep.unsqueeze(1)
        decoded_sources = self.decoder(masked_tf_rep)
        decoded_sources = pad_x_to_y(decoded_sources, waveforms)
        decoded_sources = decoded_sources.transpose(1, 2)
        outputs = torch.flatten(masked_tf_rep, start_dim=0, end_dim=1)
        # shape (batch * nsrc, nfilters, nframes)
        outputs = self.average_pool(outputs)
        outputs = outputs.transpose(1, 2)
        # shape (batch, nframes, nfilters)
        if self.use_lstm:
            outputs, _ = self.lstm(outputs)
        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
        if not self.use_lstm and self.hparams.linear["num_layers"] == 0:
            outputs = (outputs**2).sum(dim=2).unsqueeze(-1)
        outputs = self.classifier(outputs)
        
        outputs = outputs.reshape(bsz, self.n_sources, -1)
        outputs = outputs.transpose(1, 2)

        return self.activation[0](outputs), decoded_sources
