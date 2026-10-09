# The MIT License (MIT)
#
# Copyright (c) 2024-2025 CNRS
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pytest
import torch

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.separation.ToTaToNet import ToTaToNet


@pytest.mark.parametrize(
    "linear",
    [
        None,
        {"hidden_size": 8, "num_layers": 1},
        {"hidden_size": 96, "num_layers": 2},
        {"num_layers": 0},
    ],
    ids=["default", "narrow", "wide", "no_hidden_layers"],
)
def test_totatonet_linear_configuration(linear):
    pytest.importorskip("asteroid")
    pytest.importorskip("transformers")
    model = ToTaToNet(
        use_wavlm=False,
        n_sources=2,
        linear=linear,
        encoder_decoder={"n_filters": 8},
        dprnn={"n_repeats": 1, "bn_chan": 8, "hid_size": 8, "chunk_size": 50},
    )
    model.specifications = (
        Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=2.0,
            classes=["speaker#1", "speaker#2"],
        ),
        Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=2.0,
            classes=["speaker#1", "speaker#2"],
        ),
    )
    model.build()
    waveforms = torch.randn(2, 1, 32000)
    diarization, sources = model(waveforms)
    assert diarization.shape == (2, model.num_frames(32000), 2)
    assert sources.shape == (2, 32000, 2)
    assert torch.isfinite(diarization).all()
    assert torch.isfinite(sources).all()
    (diarization.square().mean() + sources.square().mean()).backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert torch.count_nonzero(model.classifier.weight.grad) > 0
