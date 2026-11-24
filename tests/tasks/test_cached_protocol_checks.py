"""Warm task construction reads small cache headers, not source annotations."""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.protocol import SpeakerDiarizationProtocol

from pyannote.audio.tasks import SpeakerDiarization


class CacheProtocol(SpeakerDiarizationProtocol):
    name = "Test.SpeakerDiarization.CachedChecks"

    def __init__(self, validation=True, classes=False):
        super().__init__()
        self.validation = validation
        self.classes = classes
        self.reads = []

    def entry(self, subset):
        annotation = Annotation()
        annotation[Segment(0, 4)] = "speaker"
        file = {
            "database": "test",
            "uri": subset,
            "audio": "fake.wav",
            "scope": "file",
            "annotation": annotation,
            "annotated": Timeline([Segment(0, 4)]),
        }
        if self.classes:
            file["classes"] = ["speaker"]
        return file

    def train_iter(self):
        self.reads.append("train")
        yield self.entry("train")

    def development_iter(self):
        self.reads.append("development")
        if not self.validation:
            raise NotImplementedError
        yield self.entry("development")


class UnavailableProtocol(CacheProtocol):
    def train_iter(self):
        raise AssertionError("warm construction must not read training data")

    def development_iter(self):
        raise AssertionError("warm construction must not read validation data")


def make_task(tmp_path, protocol):
    return SpeakerDiarization(
        protocol,
        cache=tmp_path / "task.npz",
        duration=2,
        max_speakers_per_chunk=2,
        num_workers=0,
    )


def rewrite_cache(path, transform):
    with np.load(path, allow_pickle=True) as cache:
        data = dict(cache)
    transform(data)
    np.savez(path, **data)


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("classes", [False, True])
def test_warm_construction_reads_only_headers(
    tmp_path, monkeypatch, validation, classes
):
    source = CacheProtocol(validation=validation, classes=classes)
    cold = make_task(tmp_path, source)
    assert source.reads == ["train", "development"]
    cold.prepare_data()
    cold.setup()
    with np.load(cold.cache, allow_pickle=False) as cache:
        checks = cache["protocol-checks"]
        assert checks.shape == ()
        assert checks.tolist() == (validation, True, classes)

    original = np.lib.npyio.NpzFile.__getitem__

    def headers_only(data, key):
        assert key in ("protocol", "protocol-checks")
        return original(data, key)

    # Preserve the real protocol object, even when its source is unavailable.
    unavailable = UnavailableProtocol()
    with monkeypatch.context() as patch:
        patch.setattr(np.lib.npyio.NpzFile, "__getitem__", headers_only)
        warm = make_task(tmp_path, unavailable)
        warm.prepare_data()
    assert warm.protocol is unavailable
    assert warm.has_validation is validation
    assert warm.has_scope is True
    assert warm.has_classes is classes
    warm.setup()
    for key in ("audio-metadata", "annotations-segments", "annotations-regions"):
        np.testing.assert_array_equal(warm.prepared_data[key], cold.prepared_data[key])
    if validation:
        np.testing.assert_array_equal(
            warm.prepared_data["validation"], cold.prepared_data["validation"]
        )


def test_legacy_cache_falls_back_to_source_checks(tmp_path):
    cold = make_task(tmp_path, CacheProtocol())
    cold.prepare_data()
    rewrite_cache(cold.cache, lambda data: data.pop("protocol-checks"))
    source = CacheProtocol()
    warm = make_task(tmp_path, source)
    assert source.reads == ["train", "development"]
    warm.prepare_data()
    assert source.reads == ["train", "development"]
    warm.setup()
    assert "protocol-checks" not in warm.prepared_data


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_cache_runs_source_checks(tmp_path, empty):
    if empty:
        (tmp_path / "task.npz").touch()
    source = CacheProtocol()
    task = make_task(tmp_path, source)
    assert source.reads == ["train", "development"]
    task.prepare_data()
    task.setup()
    assert "protocol-checks" in task.prepared_data


@pytest.mark.parametrize("legacy", [False, True])
def test_protocol_mismatch_rejected_before_source_reads(tmp_path, legacy):
    task = make_task(tmp_path, CacheProtocol())
    task.prepare_data()
    if legacy:
        rewrite_cache(task.cache, lambda data: data.pop("protocol-checks"))
    source = UnavailableProtocol()
    source.name = "Wrong.SpeakerDiarization.Protocol"
    with pytest.raises(ValueError, match="does not match"):
        make_task(tmp_path, source)


@pytest.mark.parametrize("empty", [False, True])
def test_disappearing_warm_cache_requires_new_task(tmp_path, empty):
    cold = make_task(tmp_path, CacheProtocol())
    cold.prepare_data()
    warm = make_task(tmp_path, UnavailableProtocol())
    if empty:
        warm.cache.write_bytes(b"")
    else:
        warm.cache.unlink()
    with pytest.raises(FileNotFoundError, match="construct a new task"):
        warm.prepare_data()
    # Normal construction still installs preprocessors/checks before rebuilding.
    rebuilt = make_task(tmp_path, CacheProtocol())
    rebuilt.prepare_data()
    rebuilt.setup()


@pytest.mark.parametrize(
    "checks",
    [
        np.array([True, True, False]),
        np.array(
            {"has_validation": True, "has_scope": True, "has_classes": False},
            dtype=object,
        ),
    ],
)
def test_malformed_checks_rejected_without_source_reads(tmp_path, checks):
    cold = make_task(tmp_path, CacheProtocol())
    cold.prepare_data()
    rewrite_cache(cold.cache, lambda data: data.update({"protocol-checks": checks}))
    with pytest.raises(ValueError):
        make_task(tmp_path, UnavailableProtocol())
