# MIT License
#
# Copyright (c) 2024- CNRS
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

# AUTHOR: Joonas Kalda (github.com/joonaskalda)

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import make_enc_dec
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

try:
    from asteroid.masknn import DPRNN
    from asteroid.utils.torch_utils import pad_x_to_y

    ASTEROID_IS_AVAILABLE = True
except ImportError:
    ASTEROID_IS_AVAILABLE = False


try:
    from transformers import AutoModel

    TRANSFORMERS_IS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_IS_AVAILABLE = False


class ToTaToNet(Model):
    """ToTaToNet joint speaker diarization and speech separation model

                        /--------------\\
    Conv1D Encoder --------+--- DPRNN --X------- Conv1D Decoder
    WavLM -- upsampling --/                 \\--- Avg pool -- Linear -- Classifier


    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        See ToTaToNet.LINEAR_DEFAULTS for default values.
    diar : dict, optional
        Keyword arguments used to initalize the average pooling in the diarization branch.
        See ToTaToNet.DIAR_DEFAULTS for default values.
    encoder_decoder : dict, optional
        Keyword arguments used to initalize the encoder and decoder.
        See ToTaToNet.ENCODER_DECODER_DEFAULTS for default values.
    dprnn : dict, optional
        Keyword arguments used to initalize the DPRNN model.
        See ToTaToNet.DPRNN_DEFAULTS for default values.
    sample_rate : int, optional
        Audio sample rate. Defaults to 16000.
    num_channels : int, optional
        Number of channels. Defaults to 1.
    task : Task, optional
        Task to perform. Defaults to None.
    n_sources : int, optional
        Number of separated sources. Defaults to 3.
    use_wavlm : bool, optional
        Whether to use the WavLM large model for feature extraction. Defaults to True.
    gradient_clip_val : float, optional
        Gradient clipping value. Required when fine-tuning the WavLM model and thus using two different optimizers.
        Defaults to 5.0.

    References
    ----------
    Joonas Kalda, Clément Pagés, Ricard Marxer, Tanel Alumäe, and Hervé Bredin.
    "PixIT: Joint Training of Speaker Diarization and Speech Separation
    from Real-world Multi-speaker Recordings"
    Odyssey 2024. https://arxiv.org/abs/2403.02288
    """

    ENCODER_DECODER_DEFAULTS = {
        "fb_name": "free",
        "kernel_size": 32,
        "n_filters": 64,
        "stride": 16,
    }
    LINEAR_DEFAULTS = {"hidden_size": 64, "num_layers": 2}
    DPRNN_DEFAULTS = {
        "n_repeats": 6,
        "bn_chan": 128,
        "hid_size": 128,
        "chunk_size": 100,
        "norm_type": "gLN",
        "mask_act": "relu",
        "rnn_type": "LSTM",
    }
    DIAR_DEFAULTS = {"frames_per_second": 125}

    def __init__(
        self,
        encoder_decoder: dict = None,
        linear: Optional[dict] = None,
        diar: Optional[dict] = None,
        dprnn: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        n_sources: int = 3,
        use_wavlm: bool = True,
        gradient_clip_val: float = 5.0,
    ):
        if not ASTEROID_IS_AVAILABLE:
            raise ImportError(
                "'asteroid' must be installed to use ToTaToNet separation. "
                "`pip install pyannote-audio[separation]` should do the trick."
            )

        if not TRANSFORMERS_IS_AVAILABLE:
            raise ImportError(
                "'transformers' must be installed to use ToTaToNet separation. "
                "`pip install pyannote-audio[separation]` should do the trick."
            )

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        dprnn = merge_dict(self.DPRNN_DEFAULTS, dprnn)
        encoder_decoder = merge_dict(self.ENCODER_DECODER_DEFAULTS, encoder_decoder)
        diar = merge_dict(self.DIAR_DEFAULTS, diar)
        self.use_wavlm = use_wavlm
        self.save_hyperparameters("encoder_decoder", "linear", "dprnn", "diar")
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

        if self.use_wavlm:
            self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-large")
            downsampling_factor = 1
            for conv_layer in self.wavlm.feature_extractor.conv_layers:
                if isinstance(conv_layer.conv, nn.Conv1d):
                    downsampling_factor *= conv_layer.conv.stride[0]
            self.wavlm_scaling = int(downsampling_factor / encoder_decoder["stride"])

            self.masker = DPRNN(
                encoder_decoder["n_filters"]
                + self.wavlm.feature_projection.projection.out_features,
                out_chan=encoder_decoder["n_filters"],
                n_src=n_sources,
                **self.hparams.dprnn,
            )
        else:
            self.masker = DPRNN(
                encoder_decoder["n_filters"],
                out_chan=encoder_decoder["n_filters"],
                n_src=n_sources,
                **self.hparams.dprnn,
            )

        # diarization can use a lower resolution than separation
        self.diarization_scaling = int(
            sample_rate / diar["frames_per_second"] / encoder_decoder["stride"]
        )
        self.average_pool = nn.AvgPool1d(
            self.diarization_scaling, stride=self.diarization_scaling
        )
        linaer_input_features = n_feats_out
        if linear["num_layers"] > 0:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [
                            linaer_input_features,
                        ]
                        + [self.hparams.linear["hidden_size"]]
                        * self.hparams.linear["num_layers"]
                    )
                ]
            )
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return 1

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            self.classifier = nn.Linear(64, self.dimension)
        else:
            self.classifier = nn.Linear(1, self.dimension)
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

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_num_frames(
            num_samples, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

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

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_receptive_field_size(
            num_frames, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

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

        equivalent_stride = (
            self.diarization_scaling * self.hparams.encoder_decoder["stride"]
        )
        equivalent_kernel_size = (
            self.diarization_scaling * self.hparams.encoder_decoder["kernel_size"]
        )

        return conv1d_receptive_field_center(
            frame, kernel_size=equivalent_kernel_size, stride=equivalent_stride
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        sources : (batch, sample, n_sources)
        """
        bsz = waveforms.shape[0]
        tf_rep = self.encoder(waveforms)
        if self.use_wavlm:
            wavlm_rep = self.wavlm(waveforms.squeeze(1)).last_hidden_state
            wavlm_rep = wavlm_rep.transpose(1, 2)
            wavlm_rep = wavlm_rep.repeat_interleave(self.wavlm_scaling, dim=-1)
            wavlm_rep = pad_x_to_y(wavlm_rep, tf_rep)
            wavlm_rep = torch.cat((tf_rep, wavlm_rep), dim=1)
            masks = self.masker(wavlm_rep)
        else:
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
        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
        if self.hparams.linear["num_layers"] == 0:
            outputs = (outputs**2).sum(dim=2).unsqueeze(-1)
        outputs = self.classifier(outputs)
        outputs = outputs.reshape(bsz, self.n_sources, -1)
        outputs = outputs.transpose(1, 2)

        return self.activation[0](outputs), decoded_sources
