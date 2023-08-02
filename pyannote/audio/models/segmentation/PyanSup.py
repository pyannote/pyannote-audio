from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.selfsup import SelfSupModel
from pyannote.audio.utils.params import merge_dict


class PyanSup(Model):
    """PyanHugg segmentation model

    Self-Supervised Model > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    
    selfsupervised : dict, optional
        Keyword arugments passed to the selfsupervised block. name and cfg are used to reconstruct the feature extractor from the dictionary of the checkpoint. Layer corresponds to the layer that serves for the feature extraction.
        Defaults to {
        "name": None,
        "layer": None,
        "cfg": None,
    }
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
    
    
    
    SSL_DEFAULTS = {
        "name": None,
        "layer": None,
        "average_layers": None,
        "average_all": False,
        "cfg": None,
    }
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    def __init__(
        self,
        ckpt: str = None,
        torchaudio_ssl: str = None,
        torchaudio_cache: str = None,
        finetune: bool = False,
        selfsupervised: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        
        selfsupervised = merge_dict(self.SSL_DEFAULTS, selfsupervised)
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        self.save_hyperparameters("lstm", "linear") #A first merge is done using the default parameters specified
        
        print("\n##################################################################")
        print("### A self-supervised model is used for the feature extraction ###")
        print("##################################################################")
        self.selfsupervised = SelfSupModel(ckpt,torchaudio_ssl,torchaudio_cache,finetune,**selfsupervised)
        selfsupervised['name'] = self.selfsupervised.model_name
        selfsupervised['cfg'] = self.selfsupervised.config
        self.save_hyperparameters("selfsupervised")
        feat_size = self.selfsupervised.feat_size
        
        print("##################################################################\n")
        
        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(feat_size, **multi_layer_lstm)

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
                        feat_size
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

        if self.specifications.powerset:
            out_features = self.specifications.num_powerset_classes
        else:
            out_features = len(self.specifications.classes)

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
        outputs = self.selfsupervised(waveforms)
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
                
        return self.activation(self.classifier(outputs))
