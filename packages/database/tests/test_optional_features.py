# Copyright (c) 2025- pyannoteAI

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest


def test_cli_help():
    pytest.importorskip("typer")
    result = subprocess.run(
        [str(Path(sys.executable).with_name("pyannote-database")), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Usage:" in result.stdout


def test_loader_entry_points():
    loaders = {
        ep.name: ep.load() for ep in entry_points(group="pyannote.database.loader")
    }
    assert {".rttm", ".uem", ".map", ".lab", ".stm"} <= loaders.keys()
    assert all(callable(loader) for loader in loaders.values())


@pytest.mark.parametrize("templated", [False, True])
def test_stm_loader(tmp_path, templated):
    pytest.importorskip("meeteval")
    from meeteval.io.seglst import SegLST

    from pyannote.database.loader import STMLoader

    path = tmp_path / "sample.stm"
    path.write_text("sample 1 speaker 0.0 1.0 hello world\n")
    loader = STMLoader(tmp_path / "{uri}.stm" if templated else path)
    transcript = loader({"uri": "sample"})
    assert isinstance(transcript, SegLST)
    assert [segment["words"] for segment in transcript] == ["hello world"]
    assert list(loader({"uri": "sample"})) == list(transcript)


def test_missing_transcription_extra():
    code = """
import sys
import warnings
sys.modules['meeteval'] = None
from pyannote.database.loader import STMLoader
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    loader = STMLoader('sample.stm')
    result = loader({'uri': 'sample'})
assert result is None
assert any('MeetEval' in str(w.message) for w in caught)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
