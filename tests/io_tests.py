import random

import torch
import torchaudio
from torch import Tensor

from pyannote.audio.core.io import Audio
from pyannote.core import Segment, SlidingWindow


def test_audio_resample():
    "Audio is correctly resampled when it isn't the correct sample rate"
    test_file = "tests/data/dev00.wav"
    info = torchaudio.info(test_file)
    old_sr = info.sample_rate
    loader = Audio(old_sr // 2)
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)
    assert sr == old_sr // 2


def test_basic_load_with_defaults():
    test_file = "tests/data/dev00.wav"
    loader = Audio()
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)


def test_correct_audio_channel():
    "When we specify an audio channel, it is chosen correctly"
    waveform = torch.rand(2, 16000 * 2)
    loader = Audio()
    wav, sr = loader({"waveform": waveform, "sample_rate": 16000, "channel": 1})
    assert torch.equal(wav, waveform[0:1])
    assert sr == 16000


def test_can_load_with_waveform():
    "We can load a raw waveform"
    waveform = torch.rand(2, 16000 * 2)
    loader = Audio()
    wav, sr = loader({"waveform": waveform, "sample_rate": 16000})
    assert isinstance(wav, Tensor)
    assert sr == 16000


def test_can_crop():
    "Cropping works when we give a Segment"
    test_file = "tests/data/dev00.wav"
    loader = Audio()
    segment = Segment(0.2, 0.7)
    wav, sr = loader.crop(test_file, segment)
    assert wav.shape[1] / sr == 0.5


def test_can_crop_waveform():
    "Cropping works on raw waveforms"
    waveform = torch.rand(1, 16000 * 2)
    loader = Audio()
    segment = Segment(0.2, 0.7)
    wav, sr = loader.crop({"waveform": waveform, "sample_rate": 16000}, segment)
    assert isinstance(wav, Tensor)
    assert sr == 16000


def test_crops_are_correct_shape():
    sr = 160001
    secs = random.randint(5, 11)
    waveform = torch.randn(1, secs * sr)
    loader = Audio()
    shape = None
    for segment in SlidingWindow(end=secs):
        wav, sr = loader.crop({"waveform": waveform, "sample_rate": sr}, segment)
        if shape is None:
            shape = wav.shape
        else:
            assert shape == wav.shape
