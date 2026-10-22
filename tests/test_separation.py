# Copyright (c) 2025- pyannoteAI

import pytest
import torch


def test_separation_masker():
    pytest.importorskip("asteroid")
    pytest.importorskip("transformers")
    from pyannote.audio.models.separation.ToTaToNet import ToTaToNet

    model = ToTaToNet(
        use_wavlm=False,
        n_sources=2,
        dprnn={"n_repeats": 1, "bn_chan": 8, "hid_size": 8, "chunk_size": 10},
    ).eval()
    with torch.inference_mode():
        encoded = model.encoder(torch.randn(1, 1, 1600))
        masks = model.masker(encoded)
    assert masks.shape == (1, 2, *encoded.shape[1:])
    assert torch.isfinite(masks).all()
