# MIT License
#
# Copyright (c) 2026- CNRS
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

"""SpeakerDiarization.get_embeddings runs the embedding model once per chunk
(with stacked speaker masks) when the backend supports it, and once per
(chunk, speaker) pair otherwise. Both groupings must produce the same
embeddings -- the masks only enter at the statistics-pooling stage, so the
frame-wise part of the model sees identical input either way."""

import types

import numpy as np
import pytest
import torch

from pyannote.audio.core.io import Audio
from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import (
    PyannoteAudioPretrainedSpeakerEmbedding,
)
from pyannote.core import SlidingWindow, SlidingWindowFeature


@pytest.fixture(scope="module")
def embedding():
    # a randomly initialised model: the equivalence under test is about
    # arithmetic, not about trained weights, so no download is needed
    torch.manual_seed(0)
    model = WeSpeakerResNet34()
    model.eval()
    return PyannoteAudioPretrainedSpeakerEmbedding(
        model, device=torch.device("cpu")
    )


def _pipeline_stub(embedding):
    # just the attributes get_embeddings touches
    return types.SimpleNamespace(
        training=False,
        _embedding=embedding,
        _audio=Audio(sample_rate=16000, mono="downmix"),
        embedding_batch_size=4,
    )


def _test_inputs():
    # 5 chunks x 20 mask frames x 3 speakers over 6 s of noise, with the
    # awkward cases: a fully inactive speaker slot, and NaN frames from
    # partial stitching
    file = {"waveform": torch.randn(1, 16000 * 6), "sample_rate": 16000}
    rng = np.random.default_rng(1)
    data = (rng.random((5, 20, 3)) > 0.5).astype(float)
    data[0, :, 2] = 0.0
    data[1, 3:5, 0] = np.nan
    binseg = SlidingWindowFeature(
        data, SlidingWindow(start=0.0, duration=2.0, step=1.0)
    )
    return file, binseg


def test_wespeaker_backend_supports_stacked_masks(embedding):
    assert embedding.supports_stacked_masks


@pytest.mark.parametrize("exclude_overlap", [False, True])
def test_stacked_masks_match_per_speaker_masks(embedding, exclude_overlap):
    file, binseg = _test_inputs()
    stub = _pipeline_stub(embedding)

    stacked = SpeakerDiarization.get_embeddings(
        stub, file, binseg, exclude_overlap=exclude_overlap
    )

    # force the per-(chunk, speaker) grouping on the same backend
    class PerPairBackend:
        supports_stacked_masks = False

        def __getattr__(self, name):
            return getattr(embedding, name)

        def __call__(self, waveforms, masks=None):
            return embedding(waveforms, masks=masks)

    stub = _pipeline_stub(PerPairBackend())
    per_pair = SpeakerDiarization.get_embeddings(
        stub, file, binseg, exclude_overlap=exclude_overlap
    )

    assert stacked.shape == per_pair.shape == (5, 3, 256)
    np.testing.assert_allclose(stacked, per_pair, atol=1e-5, rtol=1e-4)
