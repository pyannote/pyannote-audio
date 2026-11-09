from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from pyannote.core import Segment

from pyannote.audio.core.io import Audio


@pytest.mark.parametrize("kind", ["waveform", "path", "stream"])
@pytest.mark.parametrize(
    "location", ["before", "after", "ends_at_start", "starts_at_end"]
)
def test_pad_entirely_outside_audio_returns_only_requested_zeros(kind, location):
    path = Path("tests/data/dev00.wav")
    waveform, sample_rate = Audio()(path)
    duration = waveform.shape[1] / sample_rate
    segment = {
        "before": Segment(-3, -2),
        "after": Segment(duration + 2, duration + 3),
        "ends_at_start": Segment(-1, 0),
        "starts_at_end": Segment(duration, duration + 1),
    }[location]
    with ExitStack() as stack:
        if kind == "waveform":
            source = {"waveform": waveform, "sample_rate": sample_rate}
        elif kind == "stream":
            source = stack.enter_context(path.open("rb"))
        else:
            source = path
        cropped, rate = Audio().crop(source, segment, mode="pad")
        assert rate == sample_rate
        assert cropped.shape == (waveform.shape[0], sample_rate)
        assert torch.count_nonzero(cropped) == 0
        if kind == "stream":
            assert source.tell() == 0


@pytest.mark.parametrize("kind", ["waveform", "path"])
@pytest.mark.parametrize("location", ["before", "after"])
def test_pad_partial_overlap_preserves_real_audio(kind, location):
    path = Path("tests/data/dev00.wav")
    waveform, sample_rate = Audio()(path)
    duration = waveform.shape[1] / sample_rate
    n = sample_rate // 2
    if location == "before":
        segment = Segment(-0.5, 0.5)
        expected = torch.cat(
            [waveform.new_zeros(waveform.shape[0], n), waveform[:, :n]], dim=1
        )
    else:
        segment = Segment(duration - 0.5, duration + 0.5)
        expected = torch.cat(
            [waveform[:, -n:], waveform.new_zeros(waveform.shape[0], n)], dim=1
        )
    source = (
        {"waveform": waveform, "sample_rate": sample_rate}
        if kind == "waveform"
        else path
    )
    cropped, rate = Audio().crop(source, segment, mode="pad")
    assert rate == sample_rate
    torch.testing.assert_close(cropped, expected)


@pytest.mark.parametrize("kind", ["waveform", "path"])
@pytest.mark.parametrize("segment", [Segment(-3, -2), Segment(100, 101)])
def test_raise_mode_still_rejects_entirely_outside_segments(kind, segment):
    source = "tests/data/dev00.wav"
    if kind == "waveform":
        waveform, sample_rate = Audio()(source)
        source = {"waveform": waveform, "sample_rate": sample_rate}
    with pytest.raises(ValueError, match="requested chunk"):
        Audio().crop(source, segment, mode="raise")


@pytest.mark.parametrize("channel", [None, 1])
def test_zero_padding_preserves_waveform_dtype_and_resampling(channel):
    waveform = torch.ones(2, 16000, dtype=torch.float64)
    source = {"waveform": waveform, "sample_rate": 16000}
    if channel is not None:
        source["channel"] = channel
    cropped, sample_rate = Audio(mono="downmix", sample_rate=8000).crop(
        source, Segment(-3, -2), mode="pad"
    )
    assert cropped.shape == (1, 8000)
    assert sample_rate == 8000
    assert cropped.dtype == waveform.dtype
    assert cropped.device == waveform.device
    assert torch.count_nonzero(cropped) == 0


def test_file_padding_stays_on_decoder_cpu_device_with_non_cpu_default(monkeypatch):
    from torchcodec.decoders import AudioDecoder

    from pyannote.audio.core import io

    # Decode metadata before changing the default, so this checks our synthetic padding,
    # independently of whether a TorchCodec version supports non-CPU default devices.
    decoder = AudioDecoder("tests/data/dev00.wav")
    metadata = decoder.metadata
    monkeypatch.setattr(io, "AudioDecoder", lambda _: decoder)
    with torch.device("meta"):
        cropped, sample_rate = Audio().crop(
            "tests/data/dev00.wav", Segment(100, 101), mode="pad"
        )
    assert cropped.device.type == "cpu"
    assert cropped.dtype == torch.float32
    assert cropped.shape == (metadata.num_channels, sample_rate)
    assert torch.count_nonzero(cropped) == 0
