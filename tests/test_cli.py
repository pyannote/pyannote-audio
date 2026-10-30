# Copyright (c) 2025- pyannoteAI

import subprocess
import sys

import pytest


def test_cli_help():
    pytest.importorskip("typer")
    result = subprocess.run(
        [sys.executable, "-m", "pyannote.audio", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Usage:" in result.stdout
    assert "benchmark" in result.stdout


class _DummyPipeline:
    def to(self, device):
        return self


def test_cli_apply_exits_cleanly_when_pipeline_cannot_be_loaded(monkeypatch, tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from pyannote.audio import __main__ as cli

    monkeypatch.setattr(cli.Pipeline, "from_pretrained", lambda *args, **kwargs: None)

    audio = tmp_path / "audio.wav"
    audio.touch()

    result = CliRunner().invoke(cli.app, ["apply", "dummy/pipeline", str(audio)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Could not load pretrained pipeline" in result.output


def test_cli_apply_exits_cleanly_when_into_is_not_a_directory(monkeypatch, tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from pyannote.audio import __main__ as cli

    monkeypatch.setattr(
        cli.Pipeline, "from_pretrained", lambda *args, **kwargs: _DummyPipeline()
    )

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    result = CliRunner().invoke(
        cli.app, ["apply", "dummy/pipeline", str(audio_dir), "--device", "cpu"]
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "INTO must also be a directory" in result.output
