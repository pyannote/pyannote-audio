import torch
import torchaudio
from pyannote.core import Segment
from torch import Tensor

from pyannote.audio.core.io import Audio


def test_audio_resample():
    "Audio is correctly resampled when it isn't the correct sample rate"
    test_file = "tests/data/dev00.wav"
    info = torchaudio.info(test_file)
    old_sr = info.sample_rate
    loader = Audio(sample_rate=old_sr // 2, mono="downmix")
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)
    assert sr == old_sr // 2


def test_basic_load_with_defaults():
    test_file = "tests/data/dev00.wav"
    loader = Audio(mono="downmix")
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)


def test_correct_audio_channel():
    "When we specify an audio channel, it is chosen correctly"
    waveform = torch.rand(2, 16000 * 2)
    loader = Audio(mono="downmix")
    wav, sr = loader({"waveform": waveform, "sample_rate": 16000, "channel": 1})
    assert torch.equal(wav, waveform[1:2])
    assert sr == 16000


def test_can_load_with_waveform():
    "We can load a raw waveform"
    waveform = torch.rand(2, 16000 * 2)
    loader = Audio(mono="downmix")
    wav, sr = loader({"waveform": waveform, "sample_rate": 16000})
    assert isinstance(wav, Tensor)
    assert sr == 16000


def test_can_crop():
    "Cropping works when we give a Segment"
    test_file = "tests/data/dev00.wav"
    loader = Audio(mono="downmix")
    segment = Segment(0.2, 0.7)
    wav, sr = loader.crop(test_file, segment)
    assert wav.shape[1] / sr == 0.5


def test_can_crop_waveform():
    "Cropping works on raw waveforms"
    waveform = torch.rand(1, 16000 * 2)
    loader = Audio(mono="downmix")
    segment = Segment(0.2, 0.7)
    wav, sr = loader.crop({"waveform": waveform, "sample_rate": 16000}, segment)
    assert isinstance(wav, Tensor)
    assert sr == 16000


# File Like Object Tests
def test_can_load_from_file_like():
    "Load entire wav of file like"
    loader = Audio(mono="downmix")

    with open("tests/data/dev00.wav", "rb") as f:
        wav, sr = loader(f)

    assert isinstance(wav, Tensor)
    assert sr == 16000


def test_can_crop_from_file_like():
    "Load cropped sections from file like objects"
    loader = Audio(mono="downmix")

    with open("tests/data/dev00.wav", "rb") as f:
        segment = Segment(0.2, 0.7)
        wav, sr = loader.crop(f, segment)

    assert isinstance(wav, Tensor)
    assert sr == 16000
    assert wav.shape[1] == 0.5 * 16000
