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


from functools import cached_property
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.probe import probe
from pyannote.core.utils.generators import pairwise


class Branch(nn.Module):
    """Branch"""

    LSTM_DEFAULTS = {"hidden_size": 128, "num_layers": 1, "bidirectional": True}
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 1}

    def __init__(
        self, in_features, lstm: dict = None, linear: dict = None, head: int = None
    ):

        super().__init__()

        self.lstm_hparams = dict(**self.LSTM_DEFAULTS)
        if lstm is not None:
            self.lstm_hparams.update(**lstm)
        self.lstm_hparams["batch_first"] = True  # this is not negotiable

        self.linear_hparams = dict(**self.LINEAR_DEFAULTS)
        if linear is not None:
            self.linear_hparams.update(**linear)

        self.lstm = nn.ModuleList()
        one_lstm_hparams = dict(self.lstm_hparams)
        one_lstm_hparams["num_layers"] = 1
        for i in range(self.lstm_hparams["num_layers"]):
            self.lstm.append(nn.LSTM(in_features, **one_lstm_hparams))
            in_features = self.lstm_hparams["hidden_size"] * (
                2 if self.lstm_hparams["bidirectional"] else 1
            )

        lstm_out_features = in_features
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.linear_hparams["hidden_size"]]
                    * self.linear_hparams["num_layers"]
                )
            ]
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        frames : (batch, ???, ???)
        """

        outputs = frames

        for i, lstm in enumerate(self.lstm):
            if i == 0:
                outputs, hidden = lstm(outputs)
            else:
                outputs, hidden = lstm(outputs, hidden)

        if len(self.linear) > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return outputs

    @cached_property
    def out_features(self) -> int:

        if len(self.linear) > 0:
            return self.linear_hparams["hidden_size"]

        return self.lstm_hparams["hidden_size"] * (
            2 if self.lstm_hparams["bidirectional"] else 1
        )


class Trunk(Branch):
    def __init__(self, in_features, lstm: dict = None):
        no_linear = {"num_layers": 0}
        super().__init__(in_features, lstm=lstm, linear=no_linear)


class MultiPyanNet(Model):
    """PyanNet segmentation model

    SincFilterbank > Conv1 > Conv1d > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    trunk : dict, optional
    branches : dict, optional
    """

    TRUNK_DEFAULTS = {
        "lstm": {"num_layers": 3, "hidden_size": 128, "bidirectional": True},
    }

    BRANCH_DEFAULTS = {
        "lstm": {"num_layers": 1, "hidden_size": 128, "bidirectional": True},
        "linear": {"num_layers": 1, "hidden_size": 128},
    }

    BRANCHES_DEFAULTS = {
        "vad": {"head": 0, **BRANCH_DEFAULTS},
        "scd": {"head": 1, **BRANCH_DEFAULTS},
        "osd": {"head": 2, **BRANCH_DEFAULTS},
    }

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        trunk: dict = None,
        branches: dict = None,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if trunk is None:
            trunk = self.TRUNK_DEFAULTS
        self.hparams.trunk = trunk

        if branches is None:
            branches = self.BRANCHES_DEFAULTS
        self.hparams.branches = branches

        self.sincnet = SincNet(sample_rate=sample_rate)
        self.trunk = Trunk(60, **self.hparams.trunk)

    def build(self):

        probes = dict()
        self.branches = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for task_name, specifications in self.hparams.task_specifications.items():
            branch_hparams = self.hparams.branches[task_name]
            probes[task_name] = f"lstm.{branch_hparams['head']:d}"
            branch = Branch(self.trunk.out_features, **branch_hparams)
            self.branches[task_name] = branch
            self.classifiers[task_name] = nn.Linear(
                branch.out_features, len(specifications.classes)
            )

        # probe trunk
        _ = probe(self.trunk, probes)

        self.activation: dict = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : {task: (batch, frame, classes) scores} dict
        """

        frames = self.sincnet(waveforms)

        outputs: dict = self.trunk(
            rearrange(frames, "batch feature frame -> batch frame feature")
        )

        return {
            task_name: self.activation[task_name](
                self.classifiers[task_name](
                    self.branches[task_name](outputs[task_name][0])
                )
            )
            for task_name in self.hparams.task_specifications
        }
