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
from einops import rearrange, reduce
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task


class SimpleSegmentationModel(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.mfcc = MFCC(
            sample_rate=self.hparams.sample_rate,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=False,
        )

        self.lstm = nn.LSTM(
            self.mfcc.n_mfcc * self.hparams.num_channels,
            32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def build(self):
        # define task-dependent layers
        self.classifier = nn.Linear(
            32 * 2, len(self.hparams.task_specifications.classes)
        )
        self.activation = self.default_activation()

        # why do we define those layers here and not in task.setup()?
        # because, at inference time, we need those layers.

        # this is in contrast of SpeakerEmbedding.loss_func layers below
        # that are only needed during training -- we don't want them to
        # be applied at inference.

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : (batch, time, channel)

        Returns
        -------
        scores : (batch, time, classes)
        """
        # extract MFCC
        mfcc = self.mfcc(waveforms)
        # pass MFCC sequeence into the recurrent layer
        output, hidden = self.lstm(rearrange(mfcc, "b c f t -> b t (c f)"))
        # apply the final classifier to get logits
        return self.activation(self.classifier(output))


class MultiTaskSegmentationModel(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.mfcc = MFCC(
            sample_rate=self.hparams.sample_rate,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=False,
        )

        self.lstm = nn.LSTM(
            self.mfcc.n_mfcc * self.hparams.num_channels,
            32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def build(self):

        self.classifier = nn.ModuleDict(
            {
                name: nn.Linear(32 * 2, len(specifications.classes))
                for name, specifications in self.hparams.task_specifications.items()
            }
        )

        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # extract MFCC
        mfcc = self.mfcc(waveforms)
        # pass MFCC sequence into the recurrent layer
        output, hidden = self.lstm(rearrange(mfcc, "b c f t -> b t (c f)"))

        return {
            name: self.activation[name](self.classifier[name](output))
            for name in self.hparams.task_specifications
        }


class SimpleEmbeddingModel(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.mfcc = MFCC(
            sample_rate=self.hparams.sample_rate,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=False,
        )

        self.lstm = nn.LSTM(
            self.mfcc.n_mfcc * self.hparams.num_channels,
            32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # this is needed because example_output_array is needed in SpeakerEmbedding.setup
        # to automagically infer the embedding size. but example_output_array is computed
        # in Model.setup (which is called **after** Task.setup).

        # note that this is only a problem for embedding tasks.
        # we should find a way to automate this call so that the
        # end user does not forget to call it. note that this must
        # be called at the end of __init__
        if self.task is not None:
            self.task.example_output_array = self.forward(self.task.example_input_array)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : (batch, time, channel)

        Returns
        -------
        embedding : (batch, dimension)
        """

        mfcc = self.mfcc(waveforms)
        output, hidden = self.lstm(rearrange(mfcc, "b c f t -> b t (c f)"))
        # mean temporal pooling
        return reduce(output, "b t f -> b f", "mean")
