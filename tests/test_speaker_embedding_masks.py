# MIT License
#
# Copyright (c) 2023- CNRS
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

import types

import numpy as np
import pytest
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature

from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import (
    PyannoteAudioPretrainedSpeakerEmbedding,
)


@pytest.fixture()
def embedding():
    torch.manual_seed(0)
    wrapper = PyannoteAudioPretrainedSpeakerEmbedding.__new__(
        PyannoteAudioPretrainedSpeakerEmbedding
    )
    wrapper.device = torch.device("cpu")
    wrapper.model_ = WeSpeakerResNet34().eval()
    return wrapper


@pytest.mark.parametrize("exclude_overlap", [False, True])
def test_get_embeddings_same_with_and_without_speaker_masks(
    embedding, exclude_overlap
):
    rng = np.random.default_rng(0)
    data = (rng.random((7, 60, 3)) > 0.5).astype(np.float32)
    data[2, :, 1] = 0.0  # silent speaker
    data[4, :10, :] = np.nan  # partial stitching
    segmentations = SlidingWindowFeature(
        data, SlidingWindow(start=0.0, duration=1.0, step=0.5)
    )
    waveforms = torch.randn(7, 1, 16000)

    def get_embeddings(supports_speaker_masks):
        embedding.__dict__["supports_speaker_masks"] = supports_speaker_masks
        pipeline = types.SimpleNamespace(
            training=False,
            _embedding=embedding,
            _audio=types.SimpleNamespace(
                crop=lambda file, chunk, mode: (waveforms[int(chunk.start * 2)], None)
            ),
            embedding_batch_size=6,
            embedding_block_duration=None,
        )
        return SpeakerDiarization.get_embeddings(
            pipeline, {}, segmentations, exclude_overlap=exclude_overlap
        )

    np.testing.assert_allclose(
        get_embeddings(True), get_embeddings(False), rtol=1e-3, atol=1e-4
    )


def test_block_embeddings_close_to_per_chunk(embedding):
    embedding.__dict__["sample_rate"] = 16000
    embedding.__dict__["min_num_samples"] = 400
    rng = np.random.default_rng(0)
    data = (rng.random((10, 60, 3)) > 0.4).astype(np.float32)
    segmentations = SlidingWindowFeature(
        data, SlidingWindow(start=0.0, duration=1.0, step=0.5)
    )
    waveform = torch.randn(1, 16000 * 6)

    def get_embeddings(block_duration):
        pipeline = types.SimpleNamespace(
            training=False,
            _embedding=embedding,
            _audio=types.SimpleNamespace(
                crop=lambda file, chunk, mode: (
                    waveform[:, int(chunk.start * 16000) : int(chunk.end * 16000)],
                    None,
                )
            ),
            embedding_batch_size=6,
            embedding_block_duration=block_duration,
        )
        return SpeakerDiarization.get_embeddings(pipeline, {}, segmentations)

    per_chunk, one_per_block, shared = (
        get_embeddings(None),
        get_embeddings(0.4),
        get_embeddings(2.0),
    )
    assert shared.shape == per_chunk.shape
    assert np.isfinite(shared).all()
    cos = (one_per_block * per_chunk).sum(-1) / np.linalg.norm(
        one_per_block, axis=-1
    ) / np.linalg.norm(per_chunk, axis=-1)
    assert cos.min() > 0.99  # one chunk per block: same frames, only the centering differs
