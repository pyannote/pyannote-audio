# MIT License
#
# Copyright (c) 2026- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace

import pytest
import typer
import yaml
from pyannote.audio.__main__ import Pipeline as AudioPipeline
from pyannote.audio.__main__ import Metric, app
from pyannote.core import Annotation, Segment
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Integer
from typer.testing import CliRunner


class MockPipeline(Pipeline):
    """Minimal pipeline exercising the real CLI/optimizer integration."""

    def __init__(self):
        super().__init__()
        self.threshold = Integer(0, 1)

    def __call__(self, current_file, **kwargs):
        return SimpleNamespace(speaker_diarization=current_file["annotation"])

    def default_parameters(self):
        return {"threshold": 0}

    def to(self, device):
        return self


class MockProtocol:
    def development(self):
        annotation = Annotation(uri="file")
        annotation[Segment(0.0, 1.0)] = "speaker"
        yield {
            "uri": "file",
            "annotation": annotation,
            "annotated": annotation.get_timeline(),
        }


def test_optimize_cli_with_single_objective(tmp_path, monkeypatch):
    """Scalar CLI optimization remains compatible with the new Optimizer API."""
    pipeline_yml = tmp_path / "pipeline.yaml"
    pipeline_yml.write_text("pipeline: mock\n")

    mock_pipeline = MockPipeline()
    monkeypatch.setattr(
        AudioPipeline,
        "from_pretrained",
        lambda *args, **kwargs: mock_pipeline,
    )
    monkeypatch.setattr(
        "pyannote.audio.__main__.pyannote.database.registry.get_protocol",
        lambda *args, **kwargs: MockProtocol(),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
            str(pipeline_yml),
            "Debug.SpeakerDiarization.Debug",
            "--subset",
            "development",
            "--device",
            "cpu",
            "--max-iterations",
            "1",
            "--average-case",
        ],
    )

    assert result.exit_code == 0, result.output

    optimized_yml = tmp_path / (
        "pipeline.Debug.SpeakerDiarization.Debug.development.yaml"
    )
    optimized = yaml.safe_load(optimized_yml.read_text())
    assert optimized["params"] == {"threshold": 0}
    assert optimized["optimization"]["status"]["best_loss"] == 0.0


def test_optimize_cli_with_multiple_objectives(tmp_path, monkeypatch):
    """Repeated metric options produce a named Pareto-front manifest."""
    pipeline_yml = tmp_path / "pipeline.yaml"
    pipeline_yml.write_text("pipeline: mock\n")

    mock_pipeline = MockPipeline()
    monkeypatch.setattr(
        AudioPipeline,
        "from_pretrained",
        lambda *args, **kwargs: mock_pipeline,
    )
    monkeypatch.setattr(
        "pyannote.audio.__main__.pyannote.database.registry.get_protocol",
        lambda *args, **kwargs: MockProtocol(),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "optimize",
            str(pipeline_yml),
            "Debug.SpeakerDiarization.Debug",
            "--subset",
            "development",
            "--device",
            "cpu",
            "--max-iterations",
            "1",
            "--metric",
            "DiarizationPurity",
            "--metric",
            "DiarizationCoverage",
            "--average-case",
        ],
    )

    assert result.exit_code == 0, result.output

    pareto_yml = tmp_path / (
        "pipeline.Debug.SpeakerDiarization.Debug.development."
        "DiarizationPurity+DiarizationCoverage.pareto.yaml"
    )
    pareto = yaml.safe_load(pareto_yml.read_text())["optimization"]
    assert pareto["metrics"] == ["DiarizationPurity", "DiarizationCoverage"]
    assert pareto["status"]["pareto_front"] == [
        {
            "trial": 0,
            "values": {
                "DiarizationPurity": 1.0,
                "DiarizationCoverage": 1.0,
            },
            "params": {"threshold": 0},
        }
    ]


@pytest.mark.parametrize("multi_objective", [False, True])
def test_optimize_cli_with_speaker_count_error(tmp_path, monkeypatch, multi_objective):
    """Count errors respect UEM and average files equally in both optimization modes."""

    class SpeakerCountPipeline(MockPipeline):
        def __call__(self, current_file, **kwargs):
            return SimpleNamespace(speaker_diarization=current_file["hypothesis"])

    class SpeakerCountProtocol:
        def development(self):
            for index, (duration, extra_speakers) in enumerate([(1.0, 1), (10.0, 3)]):
                reference = Annotation(uri=f"file{index}")
                reference[Segment(0.0, duration)] = "speaker"
                hypothesis = reference.copy()
                for speaker in range(extra_speakers):
                    hypothesis[Segment(0.0, duration), speaker + 1] = f"extra{speaker}"
                hypothesis[Segment(duration, duration + 1.0)] = "outside_uem"
                yield {
                    "uri": reference.uri,
                    "annotation": reference,
                    "annotated": reference.get_timeline(),
                    "hypothesis": hypothesis,
                }

    pipeline_yml = tmp_path / "pipeline.yaml"
    pipeline_yml.write_text("pipeline: mock\n")
    mock_pipeline = SpeakerCountPipeline()
    monkeypatch.setattr(
        AudioPipeline, "from_pretrained", lambda *args, **kwargs: mock_pipeline
    )
    monkeypatch.setattr(
        "pyannote.audio.__main__.pyannote.database.registry.get_protocol",
        lambda *args, **kwargs: SpeakerCountProtocol(),
    )

    metrics = ["DiarizationSpeakerCountError"]
    if multi_objective:
        metrics.append("DiarizationCoverage")
    metric_options = [option for metric in metrics for option in ("--metric", metric)]
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            str(pipeline_yml),
            "Debug.SpeakerDiarization.Debug",
            "--device",
            "cpu",
            "--max-iterations",
            "1",
            "--average-case",
            *metric_options,
        ],
    )
    assert result.exit_code == 0, result.output
    assert Metric.DiarizationSpeakerCountError.direction == "minimize"

    suffix = ".pareto.yaml" if multi_objective else ".yaml"
    output = tmp_path / (
        "pipeline.Debug.SpeakerDiarization.Debug.development."
        + "+".join(metrics)
        + suffix
    )
    optimization = yaml.safe_load(output.read_text())["optimization"]
    if multi_objective:
        assert mock_pipeline.get_direction() == ("minimize", "maximize")
        assert optimization["metrics"] == metrics
        assert optimization["status"]["pareto_front"][0]["values"] == {
            "DiarizationSpeakerCountError": 2.0,
            "DiarizationCoverage": 1.0,
        }
    else:
        assert optimization["status"]["best_loss"] == 2.0


def test_speaker_count_error_requires_recent_metrics(monkeypatch):
    import pyannote.metrics.diarization as diarization_metrics

    monkeypatch.delattr(diarization_metrics, "DiarizationSpeakerCountError", raising=False)
    with pytest.raises(typer.BadParameter, match="feat/speaker-count-metrics"):
        Metric.from_str("DiarizationSpeakerCountError")
    assert Metric.from_str("DiarizationErrorRate").name == "diarization error rate"
