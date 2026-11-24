"""Regression tests for recording-level caches with multiple annotation versions."""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.protocol import SpeakerDiarizationProtocol

from pyannote.audio.tasks import SpeakerDiarization


class VersionProtocol(SpeakerDiarizationProtocol):
    name = "Test.SpeakerDiarization.Versions"

    def __init__(self, train, development=()):
        super().__init__()
        self.train_files = train
        self.development_files = development

    def train_iter(self):
        yield from self.train_files

    def development_iter(self):
        if not self.development_files:
            raise NotImplementedError
        yield from self.development_files


def entry(uri="a", level="word", speakers=("alice",), regions=((0, 4),), **kwargs):
    annotation = Annotation(uri=uri)
    for index, speaker in enumerate(speakers):
        annotation[Segment(index, index + 1)] = speaker
    return {
        "uri": uri,
        "database": "test",
        "scope": "file",
        "audio": f"{uri}.wav",
        "annotation": annotation,
        "annotated": Timeline(Segment(*region) for region in regions),
        "version": {"metadata": {"level": level}},
        **kwargs,
    }


def prepare(tmp_path, train, development=()):
    task = SpeakerDiarization(
        VersionProtocol(train, development),
        cache=tmp_path / "cache.npz",
        duration=2.0,
        max_speakers_per_chunk=3,
        num_workers=0,
    )
    task.prepare_data()
    with np.load(task.cache, allow_pickle=True) as cache:
        data = dict(cache)
    data["metadata-values"] = data["metadata-values"].item()
    return data


def test_interleaved_versions_share_recording_and_speaker_ids(tmp_path):
    data = prepare(
        tmp_path,
        [
            entry(speakers=("alice", "bob")),
            entry("b"),
            entry(level="turn", speakers=("bob", "alice")),
        ],
    )
    assert len(data["audio-path"]) == 2
    assert data["audio-annotated"].tolist() == [4, 4]
    assert data["audio-segments-ids"].tolist() == [(0, 4), (4, 5)]
    segments = data["annotations-segments"][:4]
    assert segments["file_id"].tolist() == [0, 0, 0, 0]
    assert segments["file_label_idx"].tolist() == [0, 1, 1, 0]
    levels = data["metadata-values"]["level"]
    assert (
        segments["level"].tolist()
        == [levels.index("word")] * 2 + [levels.index("turn")] * 2
    )
    assert "version_id" not in segments.dtype.names
    assert data["annotations-version-regions"].tolist() == [
        (0, levels.index("word"), 0.0, 4.0),
        (0, levels.index("turn"), 0.0, 4.0),
        (1, levels.index("word"), 0.0, 4.0),
    ]
    assert data["annotations-versions"]["file_id"].tolist() == [0, 0, 1]
    assert "version" not in data["metadata-values"]
    assert "level" not in data["audio-metadata"].dtype.names


@pytest.mark.parametrize("version", [None, {}, {"metadata": {}}])
def test_missing_level_defaults_to_mixed(tmp_path, version):
    file = entry(level="ignored")
    file["level"] = "word"  # Top-level metadata must not override the nested level.
    if version is None:
        del file["version"]
    else:
        file["version"] = version
    data = prepare(tmp_path, [file])
    assert data["metadata-values"]["level"] == ["mixed"]
    assert data["annotations-segments"]["level"].tolist() == [0]


def test_short_silent_version_regions_survive(tmp_path):
    data = prepare(
        tmp_path,
        [
            entry(regions=((0, 4), (5, 5.5))),
            entry(level="turn", speakers=(), regions=((0, 4), (5, 5.5))),
        ],
    )
    assert len(data["annotations-versions"]) == 2
    assert data["annotations-versions"]["level"].tolist() == [0, 1]
    assert data["annotations-version-regions"].tolist() == [
        (0, 0, 0.0, 4.0),
        (0, 0, 5.0, 5.5),
        (0, 1, 0.0, 4.0),
        (0, 1, 5.0, 5.5),
    ]
    assert data["audio-annotated"].tolist() == [4.0]
    assert len(data["annotations-segments"]) == 1


def test_all_silent_versions(tmp_path):
    data = prepare(tmp_path, [entry(speakers=())])
    assert len(data["annotations-segments"]) == 0
    assert len(data["annotations-versions"]) == 1
    assert data["audio-segments-ids"].tolist() == [(0, 0)]


@pytest.mark.parametrize(
    "change",
    [
        {"audio": "other.wav"},
        {"channel": 1},
        {"scope": "database"},
    ],
)
def test_incompatible_versions_rejected(tmp_path, change):
    with pytest.raises(ValueError, match="disagree on audio, channel, or scope"):
        prepare(tmp_path, [entry(), entry(level="turn", **change)])


def test_cross_subset_recording_rejected(tmp_path):
    with pytest.raises(ValueError, match="multiple subsets"):
        prepare(tmp_path, [entry()], [entry(level="turn")])


def test_different_databases_are_different_recordings(tmp_path):
    data = prepare(tmp_path, [entry(), entry(database="other")])
    assert len(data["audio-path"]) == 2


def test_equivalent_channel_policies_can_share_recording(tmp_path):
    data = prepare(tmp_path, [entry(), entry(level="turn", channel="downmix")])
    assert len(data["audio-path"]) == 1


def test_validation_sampling_counts_recording_once(tmp_path):
    data = prepare(tmp_path, [entry()], [entry("dev"), entry("dev", level="turn")])
    assert len(data["audio-path"]) == 2
    assert len(data["validation"]) == 2
    assert data["validation"]["file_id"].tolist() == [1, 1]


@pytest.mark.parametrize("scope", ["database", "global"])
def test_scoped_speaker_ids_shared_across_versions(tmp_path, scope):
    data = prepare(
        tmp_path,
        [
            entry(scope=scope, speakers=("alice", "bob")),
            entry(scope=scope, level="turn", speakers=("bob", "alice")),
        ],
    )
    segments = data["annotations-segments"]
    assert segments["database_label_idx"].tolist() == [0, 1, 1, 0]
    if scope == "global":
        assert segments["global_label_idx"].tolist() == [0, 1, 1, 0]


def test_nested_level_takes_precedence(tmp_path):
    file = entry(level="turn")
    file["level"] = "word"
    data = prepare(tmp_path, [file])
    assert data["metadata-values"]["level"] == ["turn"]


def test_file_ids_do_not_overflow(tmp_path):
    data = prepare(tmp_path, [entry(str(index)) for index in range(130)])
    assert data["annotations-versions"]["file_id"].tolist() == list(range(130))
    assert data["annotations-segments"]["file_id"].tolist() == list(range(130))
    assert data["annotations-version-regions"]["file_id"].tolist() == list(range(130))


def test_invalid_level_rejected(tmp_path):
    with pytest.raises(ValueError, match="Annotation level must be a string"):
        prepare(tmp_path, [entry(level=42)])


@pytest.mark.parametrize("level", ["word", "turn", "mixed"])
def test_duplicate_levels_rejected_even_for_silent_versions(tmp_path, level):
    with pytest.raises(
        ValueError, match=f"Recording .* has multiple versions with level '{level}'"
    ):
        prepare(tmp_path, [entry(level=level), entry(level=level, speakers=())])


def test_missing_level_conflicts_with_explicit_mixed(tmp_path):
    file = entry()
    del file["version"]
    with pytest.raises(ValueError, match="multiple versions with level 'mixed'"):
        prepare(tmp_path, [file, entry(level="mixed")])


def test_two_missing_levels_rejected(tmp_path):
    first, second = entry(), entry()
    del first["version"]
    second["version"] = {"metadata": {}}
    with pytest.raises(ValueError, match="multiple versions with level 'mixed'"):
        prepare(tmp_path, [first, second])


def test_version_without_segments_or_regions_remains_available(tmp_path):
    data = prepare(
        tmp_path,
        [entry(speakers=(), regions=()), entry(level="turn", speakers=(), regions=())],
    )
    assert data["annotations-versions"].tolist() == [(0, 0), (0, 1)]
    assert len(data["annotations-segments"]) == 0
    assert len(data["annotations-version-regions"]) == 0


@pytest.mark.parametrize("regions", [(), ((0, 3),), ((0, 1), (2, 4))])
def test_versions_with_different_annotated_regions_rejected(tmp_path, regions):
    with pytest.raises(ValueError, match="must share the same annotated regions"):
        prepare(tmp_path, [entry(), entry(level="turn", regions=regions)])


def test_equivalent_coverage_is_normalized(tmp_path):
    data = prepare(
        tmp_path, [entry(), entry(level="turn", regions=((0, 2), (1, 3), (3, 4)))]
    )
    assert data["annotations-version-regions"].tolist() == [
        (0, 0, 0.0, 4.0),
        (0, 1, 0.0, 4.0),
    ]
    assert data["audio-annotated"].tolist() == [4.0]
