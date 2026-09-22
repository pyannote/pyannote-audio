import numpy as np
import pytest
import torch

from pyannote.audio.models.embedding import WeSpeakerResNet34
from pyannote.audio.pipelines.speaker_verification import (
    PyannoteAudioPretrainedSpeakerEmbedding,
)

NUM_SAMPLES = 16000
NUM_FRAMES = 100


@pytest.fixture(scope="module")
def embedding():
    torch.manual_seed(0)
    return PyannoteAudioPretrainedSpeakerEmbedding(WeSpeakerResNet34())


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    waveforms = torch.randn(2, 1, NUM_SAMPLES)
    masks = (torch.rand(2, 3, NUM_FRAMES) > 0.5).float()
    # make sure no mask is empty
    masks[:, :, :10] = 1.0
    return waveforms, masks


def test_supports_multi_speaker_masks(embedding):
    assert embedding.supports_multi_speaker_masks


def test_multi_speaker_masks_match_one_speaker_at_a_time(embedding, batch):
    """One call with all speakers must return what several calls would return"""

    waveforms, masks = batch

    together = embedding(waveforms, masks=masks)
    assert together.shape == (2, 3, embedding.dimension)

    apart = np.stack(
        [embedding(waveforms, masks=masks[:, speaker]) for speaker in range(3)],
        axis=1,
    )
    # (2, 3, dimension)

    np.testing.assert_allclose(together, apart, rtol=1e-4, atol=1e-5)
