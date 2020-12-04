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

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.pooling import StatsPool
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.tdnn import TDNN


class XVector(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.sincnet = SincNet(sample_rate=sample_rate)

        self.frame1 = TDNN(
            context=[-2, 2],
            input_channels=60,
            output_channels=512,
            full_context=True,
        )
        self.frame2 = TDNN(
            context=[-2, 0, 2],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame3 = TDNN(
            context=[-3, 0, 3],
            input_channels=512,
            output_channels=512,
            full_context=False,
        )
        self.frame4 = TDNN(
            context=[0], input_channels=512, output_channels=512, full_context=True
        )
        self.frame5 = TDNN(
            context=[0], input_channels=512, output_channels=1500, full_context=True
        )

        self.stats_pool = StatsPool()

        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)

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
        waveforms : (batch, channel, sample)

        """

        outputs = self.sincnet(waveforms)
        outputs = self.frame1(outputs)
        outputs = self.frame2(outputs)
        outputs = self.frame3(outputs)
        outputs = self.frame4(outputs)
        outputs = self.frame5(outputs)
        outputs = self.stats_pool(outputs)
        outputs = self.segment6(F.relu(outputs))
        return self.segment7(F.relu(outputs))
