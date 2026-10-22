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
