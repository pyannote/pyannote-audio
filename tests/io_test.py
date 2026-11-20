import random

import pytest
import torch
from pyannote.core import Segment
from torch import Tensor
from torchcodec.decoders import AudioDecoder, AudioStreamMetadata

from pyannote.audio.core.io import Audio, Channel


def test_audio_resample():
    "Audio is correctly resampled when it isn't the correct sample rate"
    test_file = "tests/data/dev00.wav"
    metadata: AudioStreamMetadata = AudioDecoder(test_file).metadata
    old_sr = metadata.sample_rate
    loader = Audio(sample_rate=old_sr // 2)
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)
    assert sr == old_sr // 2


def test_basic_load_with_defaults():
    test_file = "tests/data/dev00.wav"
    loader = Audio()
    wav, _sr = loader(test_file)
    assert isinstance(wav, Tensor)


def test_correct_audio_channel():
    "When we specify an audio channel, it is chosen correctly"
    waveform = torch.rand(2, 16000 * 2)
    loader = Audio()
    wav, sr = loader({"waveform": waveform, "sample_rate": 16000, "channel": 1})
    assert torch.equal(wav, waveform[1:2])
    assert sr == 16000


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_random_audio_channel(monkeypatch, crop, num_channels):
    waveform = torch.arange(num_channels).float().unsqueeze(1).repeat(1, 16000)
    file = {"waveform": waveform, "sample_rate": 16000, "channel": "random"}
    loader = Audio(sample_rate=8000)
    rng = random.Random(42)
    monkeypatch.setattr("pyannote.audio.core.io.random.randint", rng.randint)
    expected_rng = random.Random(42)
    for _ in range(5):
        channel = expected_rng.randint(0, num_channels - 1)
        selected = {**file, "channel": channel}
        if crop:
            actual, sr = loader.crop(file, Segment(0.25, 0.75))
            expected, _ = loader.crop(selected, Segment(0.25, 0.75))
        else:
            actual, sr = loader(file)
            expected, _ = loader(selected)
        assert sr == 8000
        assert actual.shape[0] == 1
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("as_enum", [False, True])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_all_audio_channels(crop, as_enum, num_channels):
    waveform = torch.randn(num_channels, 16000)
    file = {
        "waveform": waveform,
        "sample_rate": 16000,
        "channel": Channel.ALL if as_enum else "all",
    }
    loader = Audio(sample_rate=8000)
    if crop:
        actual, sr = loader.crop(file, Segment(0.25, 0.75))
    else:
        actual, sr = loader(file)

    # Each output channel must match independently loading and resampling that
    # channel, preserving their original order.
    expected = []
    for channel in range(num_channels):
        selected = {**file, "channel": channel}
        if crop:
            single, _ = loader.crop(selected, Segment(0.25, 0.75))
        else:
            single, _ = loader(selected)
        expected.append(single)
    assert sr == 8000
    assert actual.shape[0] == num_channels
    torch.testing.assert_close(actual, torch.cat(expected))


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("as_enum", [False, True])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_explicit_downmix(crop, as_enum, num_channels, monkeypatch):
    waveform = torch.randn(num_channels, 16000)
    file = {
        "waveform": waveform,
        "sample_rate": 16000,
        "channel": Channel.DOWNMIX if as_enum else "downmix",
    }
    averaged = {"waveform": waveform.mean(dim=0, keepdim=True), "sample_rate": 16000}
    loader = Audio(sample_rate=8000)

    def unexpected_random(*args):
        pytest.fail("Explicit downmix must not randomly select a channel")

    monkeypatch.setattr("pyannote.audio.core.io.random.randint", unexpected_random)
    if crop:
        actual, sr = loader.crop(file, Segment(0.25, 0.75))
        expected, _ = loader.crop(averaged, Segment(0.25, 0.75))
    else:
        actual, sr = loader(file)
        expected, _ = loader(averaged)
    assert sr == 8000
    assert actual.shape[0] == 1
    torch.testing.assert_close(actual, expected)


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


def test_can_crop_waveform_up_to_its_end():
    "Cropping a raw waveform up to its very last sample does not raise"
    waveform = torch.rand(1, 16000 * 2)
    loader = Audio()
    segment = Segment(1.5, 2.0)
    wav, _sr = loader.crop({"waveform": waveform, "sample_rate": 16000}, segment)
    assert torch.equal(wav, waveform[:, 24000:])


# File Like Object Tests
def test_can_load_from_file_like():
    "Load entire wav of file like"
    loader = Audio()

    with open("tests/data/dev00.wav", "rb") as f:
        wav, sr = loader(f)

    assert isinstance(wav, Tensor)
    assert sr == 16000


def test_can_crop_from_file_like():
    "Load cropped sections from file like objects"
    loader = Audio()

    with open("tests/data/dev00.wav", "rb") as f:
        segment = Segment(0.2, 0.7)
        wav, sr = loader.crop(f, segment)

    assert isinstance(wav, Tensor)
    assert sr == 16000
    assert wav.shape[1] == 0.5 * 16000


def test_get_decoder_uses_wav_decoder_for_wav():
    from pyannote.audio.core.io import get_decoder

    assert type(get_decoder("tests/data/dev00.wav")).__name__ == "WavDecoder"


def test_get_decoder_falls_back_for_non_wav(tmp_path):
    "Non-WAV files uses the FFmpeg-based AudioDecoder"
    from torchcodec.encoders import AudioEncoder
    from pyannote.audio.core.io import get_decoder

    wav, sr = Audio()("tests/data/dev00.wav")
    flac = tmp_path / "dev00.flac"
    AudioEncoder(wav, sample_rate=sr).to_file(str(flac))
    decoder = get_decoder(str(flac))
    assert type(decoder).__name__ == "AudioDecoder"
    torch.testing.assert_close(decoder.get_all_samples().data, wav, atol=1e-4, rtol=0)


@pytest.mark.parametrize("file_like", [False, True])
def test_wav_decoder_matches_audio_decoder(monkeypatch, file_like):
    "WavDecoder (fast path) returns the same samples as the FFmpeg-based AudioDecoder"
    test_file = "tests/data/dev00.wav"
    loader = Audio()
    segment = Segment(3.21, 7.89)

    def load(path):
        audio = open(path, "rb") if file_like else path
        wav, sr = loader(audio)
        audio = open(path, "rb") if file_like else path
        chunk, _ = loader.crop(audio, segment)
        return wav, chunk, sr, loader.get_duration(path)

    fast = load(test_file)
    monkeypatch.setattr("pyannote.audio.core.io.WavDecoder", None)
    slow = load(test_file)

    torch.testing.assert_close(fast[0], slow[0])
    torch.testing.assert_close(fast[1], slow[1])
    assert fast[2] == slow[2]
    assert fast[3] == pytest.approx(slow[3])
