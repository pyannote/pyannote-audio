"""``SpeakerDiarization._gpu_empty_cache()`` must be a no-op on CPU-only hosts.

It is called twice from every ``apply()``. It used to fall through to
``torch.mps.empty_cache()`` whenever CUDA was absent, which raises
``RuntimeError: Cannot execute emptyCache() without MPS backend`` on a CPU-only
Linux host, so in-process diarization on a CPU worker always crashed.
"""

from __future__ import annotations

import pytest
import torch

from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization


def _record(calls: list[str], name: str):
    return lambda: calls.append(name)


def test_cpu_only_host_releases_nothing_and_does_not_raise(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", _record(calls, "cuda"))

    def _mps_empty_cache_without_backend():
        calls.append("mps")
        raise RuntimeError("Cannot execute emptyCache() without MPS backend")

    monkeypatch.setattr(torch.mps, "empty_cache", _mps_empty_cache_without_backend)

    SpeakerDiarization._gpu_empty_cache()

    assert calls == []


@pytest.mark.skipif(
    torch.cuda.is_available() or torch.backends.mps.is_available(),
    reason="exercises the real torch call on a host with no GPU backend",
)
def test_real_torch_on_cpu_only_host_does_not_raise():
    SpeakerDiarization._gpu_empty_cache()


def test_cuda_host_releases_cuda_cache(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", _record(calls, "cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", _record(calls, "mps"))

    SpeakerDiarization._gpu_empty_cache()

    assert calls == ["cuda"]


def test_mps_host_releases_mps_cache(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", _record(calls, "cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", _record(calls, "mps"))

    SpeakerDiarization._gpu_empty_cache()

    assert calls == ["mps"]
