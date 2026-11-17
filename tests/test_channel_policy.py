"""Channel policies behave identically with raw, preloaded, and cached audio."""

import random
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyannote.core import Annotation, Segment
from pyannote.database.protocol import SpeakerVerificationProtocol
from pyannote.database.protocol.protocol import ProtocolFile

from pyannote.audio.core.io import Audio, Channel
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.tasks.embedding.mixins import (
    SupervisedRepresentationLearningTaskMixin,
)
from pyannote.audio.utils.preprocessors import Waveform


@pytest.fixture
def multichannel_file(tmp_path):
    samples = np.tile(np.array([4096, 8192, 24576], dtype="<i2"), (16000, 1))
    path = tmp_path / "channels.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(3)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    return {"audio": path}


@pytest.mark.parametrize("channel", [None, 0, 2, *Channel, "all", "downmix", "random"])
def test_preloading_preserves_policy(multichannel_file, channel, monkeypatch):
    file = {**multichannel_file, "channel": channel}
    pipeline = Pipeline()
    raw = pipeline.prepare_one(dict(file), preload=False)
    preloaded = pipeline.prepare_one(dict(file), preload=True)
    assert preloaded["waveform"].shape == (3, 16000)
    assert preloaded["channel"] == channel
    loader = Audio(sample_rate=8000)
    for cropped in (False, True):
        # Identical random state must give identical selections with preloading.
        outputs = []
        for source in (raw, preloaded):
            monkeypatch.setattr(
                "pyannote.audio.core.io.random.randint", random.Random(7).randint
            )
            result = (
                loader.crop(source, Segment(0.25, 0.75)) if cropped else loader(source)
            )
            outputs.append(result[0])
        torch.testing.assert_close(*outputs)


@pytest.mark.parametrize("channel", [2, Channel.ALL, Channel.DOWNMIX])
def test_waveform_preprocessor_preserves_raw_channels(multichannel_file, channel):
    file = ProtocolFile(
        {**multichannel_file, "channel": channel, "sample_rate": 16000},
        lazy={"waveform": Waveform()},
    )
    assert file["waveform"].shape == (3, 16000)
    actual, _ = Audio()(file)
    expected, _ = Audio()({**multichannel_file, "channel": channel})
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_missing_channel_defaults_to_downmix(crop, explicit_none):
    waveform = torch.rand(3, 16000)
    file = {"waveform": waveform, "sample_rate": 16000}
    if explicit_none:
        file["channel"] = None
    loader = Audio()
    actual, _ = loader.crop(file, Segment(0.25, 0.75)) if crop else loader(file)
    expected = waveform.mean(dim=0, keepdim=True)
    if crop:
        expected = expected[:, 4000:12000]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("channel", [-1, -2, True, 1.5, "invalid", 3])
def test_invalid_channel_rejected(channel):
    with pytest.raises(ValueError, match="channel"):
        Audio()(
            {"waveform": torch.ones(3, 16), "sample_rate": 16000, "channel": channel}
        )


@pytest.mark.parametrize("channel", [0, Channel.RANDOM, Channel.ALL, Channel.DOWNMIX])
def test_embedding_training_keeps_channel(multichannel_file, channel):
    annotation = Annotation()
    annotation[Segment(0, 1)] = "speaker"
    file = {
        **multichannel_file,
        "uri": "test",
        "channel": channel,
        "annotation": annotation,
    }
    task = SimpleNamespace(
        protocol=SimpleNamespace(name="test", train=lambda: iter([file])),
        min_duration=0.5,
        duration=1,
    )
    SupervisedRepresentationLearningTaskMixin.setup(task)
    assert task._train["speaker"][0]["channel"] == channel


@pytest.mark.parametrize("channel", ["random", Channel.RANDOM])
@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_embedding_validation_supports_random(
    multichannel_file, channel, duration, monkeypatch
):
    file = {**multichannel_file, "channel": channel}
    task = SimpleNamespace(
        protocol=SpeakerVerificationProtocol(),
        prepared_data={
            "validation": [{"file1": file, "file2": file, "reference": True}]
        },
        model=SimpleNamespace(audio=Audio(sample_rate=16000)),
        duration=duration,
    )
    selections = iter([1, 2, 2, 0])
    calls = []

    def select_channel(low, high):
        assert (low, high) == (0, 2)
        selected = next(selections)
        calls.append(selected)
        return selected

    monkeypatch.setattr("pyannote.audio.core.io.random.randint", select_channel)
    for expected_values in [(0.25, 0.75), (0.75, 0.125)]:
        sample = SupervisedRepresentationLearningTaskMixin.val__getitem__(task, 0)
        for index, value in enumerate(expected_values, start=1):
            torch.testing.assert_close(
                sample[f"X{index}"], torch.full((1, int(duration * 16000)), value)
            )
        assert sample["y"] is True
    assert calls == [1, 2, 2, 0]
