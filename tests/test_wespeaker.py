# Copyright (c) 2025- pyannoteAI

import torch

from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34


def test_fbank_only():
    model = WeSpeakerResNet34().eval()
    waveforms = torch.randn(1, 1, 16000)
    with torch.inference_mode():
        expected = model.compute_fbank(waveforms)
        model.fbank_only = True
        features = model(waveforms)
        torch.testing.assert_close(features, expected)
        model.fbank_only = False
        embeddings = model(waveforms)
    assert embeddings.shape == (1, 256)
