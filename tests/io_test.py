import pytest
import torch
from pyannote.core import Segment
from torch import Tensor
from torchcodec.decoders import AudioDecoder, AudioStreamMetadata

from pyannote.audio.core.io import Audio


def test_audio_resample():
    "Audio is correctly resampled when it isn't the correct sample rate"
    test_file = "tests/data/dev00.wav"
    metadata: AudioStreamMetadata = AudioDecoder(test_file).metadata
    old_sr = metadata.sample_rate
    loader = Audio(sample_rate=old_sr // 2, mono="downmix")
    wav, sr = loader(test_file)
    assert isinstance(wav, Tensor)
    assert sr == old_sr // 2


def test_basic_load_with_defaults():
    test_file = "tests/data/dev00.wav"
    loader = Audio(mono="downmix")
    wav, _sr = loader(test_file)
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


@pytest.mark.parametrize("segment", [Segment(0.0, 1.0), Segment(2.5, 3.5)])
def test_crop_mp3_matches_full_load(segment):
    "Cropping an MP3 file counts time from its first decoded sample"
    # this MP3's first sample is timestamped 1105/16000 s (encoder delay)
    test_file = "tests/data/dev00.mp3"
    loader = Audio(mono="downmix")
    waveform, sr = loader(test_file)
    wav, _sr = loader.crop(test_file, segment)
    start = round(segment.start * sr)
    assert torch.equal(wav, waveform[:, start : start + sr])


def test_crop_ignores_header_start_of_other_codecs(monkeypatch):
    "Cropping counts from the header start only for MP3, MP2 and AAC"
    import pyannote.audio.core.io as io

    class Decoder(io.AudioDecoder):
        # with FFmpeg 8.1, Opus in Matroska can report a positive header start
        # while its decoded audio starts at 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metadata.begin_stream_seconds_from_header = 0.032

    monkeypatch.setattr(io, "AudioDecoder", Decoder)
    test_file = "tests/data/dev00.wav"
    loader = Audio(mono="downmix")
    waveform, sr = loader(test_file)
    wav, _sr = loader.crop(test_file, Segment(1.0, 2.0))
    assert torch.equal(wav, waveform[:, sr : 2 * sr])


def test_can_crop_waveform_up_to_its_end():
    "Cropping a raw waveform up to its very last sample does not raise"
    waveform = torch.rand(1, 16000 * 2)
    loader = Audio(mono="downmix")
    segment = Segment(1.5, 2.0)
    wav, _sr = loader.crop({"waveform": waveform, "sample_rate": 16000}, segment)
    assert torch.equal(wav, waveform[:, 24000:])


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
