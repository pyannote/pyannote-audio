import sys
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from pyannote.audio.core.io import Audio, Channel
from pyannote.audio.pipelines.pyannoteai.local import Local
from pyannote.audio.pipelines.pyannoteai.sdk import SDK


@pytest.fixture(params=["local", "sdk"])
def wrapper(request, monkeypatch):
    backend = Mock()
    if request.param == "local":
        backend.diarize.return_value = {"diarization": [], "exclusive_diarization": []}
        monkeypatch.setitem(
            sys.modules,
            "pyannoteai.local",
            SimpleNamespace(Pipeline=lambda token: backend),
        )
        pipeline = Local(token="test")
        payload = backend.diarize
    else:
        backend.upload.return_value = "media://test"
        backend.diarize.return_value = "test-job"
        backend.retrieve.return_value = {
            "output": {"diarization": [], "exclusiveDiarization": []}
        }
        monkeypatch.setattr(
            "pyannote.audio.pipelines.pyannoteai.sdk.Client", lambda token: backend
        )
        pipeline = SDK(token="test")
        payload = backend.upload
    return pipeline, backend, payload


@pytest.fixture
def audio_source(tmp_path):
    waveform = torch.tensor([0.125, 0.25, 0.75]).unsqueeze(1).repeat(1, 16000)
    path = tmp_path / "channels.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(3)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes((waveform.T.numpy() * 32768).astype("<i2").tobytes())
    return path, waveform


@pytest.mark.parametrize("source", ["path", "waveform", "preloaded"])
@pytest.mark.parametrize("channel", [None, 0, 2, Channel.DOWNMIX, Channel.RANDOM])
def test_wrappers_apply_selection(wrapper, audio_source, source, channel, monkeypatch):
    pipeline, _, payload = wrapper
    path, waveform = audio_source
    file = {"audio": path, "channel": channel}
    if source == "waveform":
        # Audio must prefer this waveform, rather than reread the path.
        file.update(audio="missing.wav", waveform=waveform, sample_rate=16000)
    monkeypatch.setattr("pyannote.audio.core.io.random.randint", lambda low, high: 2)
    pipeline(file, preload=source == "preloaded", num_speakers=2)
    processed = payload.call_args.args[0]
    expected, _ = Audio()(
        {"waveform": waveform, "sample_rate": 16000, "channel": channel}
    )
    assert set(processed) == {"waveform", "sample_rate"}
    np.testing.assert_allclose(processed["waveform"], expected.numpy())
    assert processed["sample_rate"] == 16000
    assert processed["waveform"].shape == (1, 16000)


@pytest.mark.parametrize("source", ["path", "waveform", "preloaded"])
def test_wrappers_reject_multichannel_all(wrapper, audio_source, source):
    pipeline, backend, _ = wrapper
    path, waveform = audio_source
    file = {"audio": path, "channel": Channel.ALL}
    if source == "waveform":
        file.update(waveform=waveform, sample_rate=16000)
    with pytest.raises(ValueError, match="require mono audio"):
        pipeline(file, preload=source == "preloaded")
    backend.diarize.assert_not_called()
    backend.upload.assert_not_called()


def test_wrappers_accept_all_for_mono(wrapper):
    pipeline, _, payload = wrapper
    waveform = torch.ones(1, 16000)
    pipeline({"waveform": waveform, "sample_rate": 16000, "channel": Channel.ALL})
    np.testing.assert_array_equal(
        payload.call_args.args[0]["waveform"], waveform.numpy()
    )
