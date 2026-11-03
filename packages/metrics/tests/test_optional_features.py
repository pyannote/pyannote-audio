"""Smoke tests for the CLI, plotting, and transcription extras."""

import subprocess
import sys

import numpy as np
import pytest


def test_cli_help():
    pytest.importorskip("docopt")
    pytest.importorskip("tabulate")
    result = subprocess.run(
        [sys.executable, "-m", "pyannote.metrics.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Usage:" in result.stdout


def test_score_distribution_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from pyannote.metrics.plot.binary_classification import plot_distributions

    target = tmp_path / "scores"
    plot_distributions(
        np.array([False, False, True, True]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        str(target),
    )
    assert target.with_suffix(".scores.png").stat().st_size > 0
    assert target.with_suffix(".scores.eps").stat().st_size > 0


def test_word_error_rate():
    pytest.importorskip("meeteval")
    from meeteval.io.seglst import SegLST

    from pyannote.metrics.transcription import WordErrorRate

    reference = SegLST(
        [{"session_id": "test", "start_time": 0, "end_time": 1, "words": "hello world"}]
    )
    hypothesis = SegLST(
        [{"session_id": "test", "start_time": 0, "end_time": 1, "words": "hello there"}]
    )
    assert WordErrorRate()(reference, hypothesis) == pytest.approx(0.5)


def test_missing_transcription_extra():
    code = """
import sys
sys.modules['meeteval'] = None
from pyannote.metrics.transcription import WordErrorRate
try:
    WordErrorRate()
except ImportError as error:
    assert 'transcription' in str(error)
else:
    raise AssertionError('Missing extra must produce an actionable ImportError')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "command", ["detection", "segmentation", "diarization", "identification"]
)
def test_cli_reports(command, monkeypatch, capsys):
    pytest.importorskip("docopt")
    pytest.importorskip("tabulate")
    from types import SimpleNamespace

    from pyannote.core import Annotation, Segment, Timeline

    from pyannote.metrics import cli
    from pyannote.metrics.base import BaseMetric

    reference = Annotation(uri="test")
    reference[Segment(0, 2)] = "speaker"
    reference[Segment(3, 5)] = "speaker"
    item = {
        "uri": "test",
        "annotation": reference,
        "annotated": Timeline([Segment(0, 6)]),
    }
    protocol = SimpleNamespace(test=lambda: iter([item]))
    reported = []
    original_report = BaseMetric.report

    def report_once(metric, *args, **kwargs):
        assert all(metric is not previous for previous in reported)
        reported.append(metric)
        return original_report(metric, *args, **kwargs)

    monkeypatch.setattr(BaseMetric, "report", report_once)
    getattr(cli, command)(protocol, "test", {"test": reference})
    output = capsys.readouterr().out
    assert command.capitalize() in output
    assert "TOTAL" in output
    assert "test" in output
    assert len(reported) == (4 if command in {"detection", "segmentation"} else 3)


@pytest.mark.parametrize("optimized", [False, True])
def test_spotting_input_validation(optimized):
    pytest.importorskip("docopt")
    pytest.importorskip("tabulate")
    code = """
from types import SimpleNamespace
from pyannote.core import Segment
from pyannote.metrics.cli import spotting

trial = {'model_id': 'speaker', 'uri': 'test', 'try_with': Segment(1, 3)}
protocol = SimpleNamespace(test_trial=lambda: iter([trial]))
valid = {'model_id': 'speaker', 'uri': 'test', 'scores': [(1, 0.5)]}
for update, message in [
    ({'model_id': 'other'}, 'target mismatch'),
    ({'uri': 'other'}, 'file mismatch'),
    ({'scores': []}, 'empty list of scores'),
    ({'scores': [(0, 0.5)]}, 'incorrect timestamp'),
]:
    try:
        spotting(protocol, 'test', [1], [valid | update], 'unused')
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f'Unexpected error: {error}') from error
    else:
        raise AssertionError(f'Missing validation: {message}')
"""
    subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", code], check=True
    )
