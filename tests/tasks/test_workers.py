# Copyright (c) 2025- pyannoteAI

import multiprocessing
import sys
import warnings

import pytest
from pyannote.database import FileFinder, registry

from pyannote.audio.tasks import VoiceActivityDetection


@pytest.mark.parametrize("start_method", ["fork", "spawn", "forkserver", None])
@pytest.mark.parametrize("requested_workers", [0, 2, None])
def test_worker_start_method(monkeypatch, start_method, requested_workers):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        multiprocessing, "get_start_method", lambda allow_none: start_method
    )
    monkeypatch.setattr(
        multiprocessing,
        "get_all_start_methods",
        lambda: ["forkserver", "fork", "spawn"],
    )
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 4)
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    fallback = requested_workers != 0 and start_method != "fork"
    with warnings.catch_warnings(record=True) as caught:
        task = VoiceActivityDetection(protocol, num_workers=requested_workers)
    if fallback:
        assert task.num_workers == 0
        assert any("cannot be pickled" in str(w.message) for w in caught)
    else:
        assert task.num_workers == (
            2 if requested_workers is None else requested_workers
        )
        assert not any("cannot be pickled" in str(w.message) for w in caught)
