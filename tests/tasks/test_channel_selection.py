import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.protocol import SpeakerDiarizationProtocol

from pyannote.audio.core import io
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.tasks import SpeakerDiarization


class ChannelProtocol(SpeakerDiarizationProtocol):
    name = "Test.SpeakerDiarization.Channels"

    def __init__(self, audio_path, channels, validation_channels=None):
        super().__init__()
        self.audio_path = audio_path
        self.channels = channels
        self.validation_channels = (
            channels if validation_channels is None else validation_channels
        )

    def files(self, subset):
        channels = self.channels if subset == "train" else self.validation_channels
        for index, channel in enumerate(channels):
            annotation = Annotation()
            annotation[Segment(0, 1)] = "speaker"
            file = {
                "uri": f"{subset}-{index}",
                "database": f"Database{index}",
                "subset": subset,
                "scope": "file",
                "audio": self.audio_path,
                "annotation": annotation,
                "annotated": Timeline([Segment(0, 1)]),
            }
            if channel is not None:
                file["channel"] = channel
            yield file

    def train_iter(self):
        yield from self.files("train")

    def development_iter(self):
        yield from self.files("development")


@pytest.mark.parametrize(
    "channels",
    [
        (0,),
        (2,),
        (None,),
        (None, 2, 0),
        ("random",),
        ("all",),
        (0, "random", None, 2, "all"),
    ],
)
@pytest.mark.parametrize("stage", ["train", "development"])
def test_cached_channel_selection(tmp_path, monkeypatch, channels, stage):
    # Mock only decoding: exercise the real crop, channel selection, and downmix.
    # Distinct constant channels make selected and averaged audio distinguishable.
    waveform = torch.tensor([0.125, 0.25, 0.75]).unsqueeze(1).repeat(1, 16000)
    audio_path = tmp_path / "multichannel.wav"
    audio_path.touch()

    class Decoder:
        def __init__(self, path):
            assert str(path) == str(audio_path)
            self.metadata = SimpleNamespace(
                sample_rate=16000, duration_seconds_from_header=1.0
            )

        def get_samples_played_in_range(self, start, end):
            return SimpleNamespace(
                data=waveform[:, round(start * 16000) : round(end * 16000)],
                sample_rate=16000,
            )

    monkeypatch.setattr(io, "AudioDecoder", Decoder, raising=False)
    monkeypatch.setattr(io, "TORCHCODEC_AVAILABLE", True)
    validation_channels = tuple(0 if c == "random" else c for c in channels)
    task = SpeakerDiarization(
        ChannelProtocol(audio_path, channels, validation_channels),
        cache=tmp_path / "task.npz",
        duration=1.0,
        max_speakers_per_chunk=2,
        num_workers=0,
    )
    task.prepare_data()
    task.setup()
    task.model = SimpleSegmentationModel()

    expected_channels = channels if stage == "train" else validation_channels
    for index, channel in enumerate(expected_channels):
        file_id = index + (len(channels) if stage == "development" else 0)
        file = task.get_file(file_id)
        if channel is None:
            assert "channel" not in file
        else:
            assert file["channel"] == channel

        selections = []

        def select_channel(low, high):
            assert (low, high) == (0, 2)
            selected = (len(selections) + 1) % 3
            selections.append(selected)
            return selected

        monkeypatch.setattr(io.random, "randint", select_channel)
        if stage == "train":
            # Select each source database in a compound protocol deterministically.
            samples = task.train__iter__helper(
                random.Random(0), database=f"Database{index}"
            )
            sample = next(samples)
        else:
            sample = task.val__getitem__(index)

        if channel == "random":
            torch.testing.assert_close(sample["X"], waveform[1:2])
            torch.testing.assert_close(next(samples)["X"], waveform[2:3])
            assert selections == [1, 2]
            continue

        assert not selections
        if channel == "all":
            expected = waveform
        elif channel is None:
            expected = waveform.mean(dim=0, keepdim=True)
        else:
            expected = waveform[channel : channel + 1]
        torch.testing.assert_close(sample["X"], expected)
        torch.testing.assert_close(
            task.collate_X([sample, sample]), torch.stack([expected, expected])
        )
        if stage == "development":
            torch.testing.assert_close(task.val__getitem__(index)["X"], expected)
            assert not selections


def make_task(tmp_path, validation_channels=(0,)):
    return SpeakerDiarization(
        ChannelProtocol(tmp_path / "audio.wav", ("random",), validation_channels),
        cache=tmp_path / "task.npz",
        duration=1.0,
        max_speakers_per_chunk=2,
        num_workers=0,
    )


def test_random_validation_rejected_during_preparation(tmp_path):
    task = make_task(tmp_path, validation_channels=("random",))
    with pytest.raises(ValueError, match="only supported for training"):
        task.prepare_data()


@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_cached_random_channels_rejected(tmp_path, legacy):
    task = make_task(tmp_path)
    task.prepare_data()
    with np.load(task.cache, allow_pickle=True) as archive:
        cached = dict(archive)
    if legacy:
        # Older caches stored "random" as a category index, indistinguishable
        # from a literal channel number in a mixed protocol.
        cached["audio-metadata"]["channel"][0] = 0
        message = "Remove the cache"
    else:
        cached["audio-metadata"]["channel"][1] = -2
        message = "only supported for training"
    np.savez(task.cache, **cached)
    with pytest.raises(ValueError, match=message):
        task.setup()


def test_legacy_all_channels_cache_rejected(tmp_path):
    task = make_task(tmp_path, validation_channels=("all",))
    task.prepare_data()
    with np.load(task.cache, allow_pickle=True) as archive:
        cached = dict(archive)
    cached["audio-metadata"]["channel"][1] = 1
    np.savez(task.cache, **cached)
    with pytest.raises(ValueError, match="Remove the cache"):
        task.setup()
