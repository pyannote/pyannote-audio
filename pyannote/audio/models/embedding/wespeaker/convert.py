# MIT License
#
# Copyright (c) 2023 CNRS
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

# Script used to convert from WeSpeaker to pyannote.audio

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

import pyannote.audio.models.embedding.wespeaker as wespeaker
from pyannote.audio import Model
from pyannote.audio.core.task import Problem, Resolution, Specifications

wespeaker_checkpoint_dir = sys.argv[1]  # /path/to/wespeaker_cnceleb-resnet34-LM

wespeaker_checkpoint = Path(wespeaker_checkpoint_dir) / "wespeaker.pt"

depth = Path(wespeaker_checkpoint_dir).parts[-1].split("-")[-2][6:]  # '34'
Klass = getattr(wespeaker, f"WeSpeakerResNet{depth}")  # WeSpeakerResNet34

duration = 5.0  # whatever
specifications = Specifications(
    problem=Problem.REPRESENTATION, resolution=Resolution.CHUNK, duration=duration
)

state_dict = torch.load(wespeaker_checkpoint, map_location=torch.device("cpu"))
state_dict.pop("projection.weight")

model = Klass()
model.resnet.load_state_dict(state_dict, strict=True)
model.specifications = specifications

checkpoint = {"state_dict": model.state_dict()}
model.on_save_checkpoint(checkpoint)
checkpoint["pytorch-lightning_version"] = pl.__version__

pyannote_checkpoint = Path(wespeaker_checkpoint_dir) / "pytorch_model.bin"
torch.save(checkpoint, pyannote_checkpoint)

model = Model.from_pretrained(pyannote_checkpoint)
print(model)
