# Copyright (c) 2026- pyannoteAI

from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from pyannote.core import Annotation, Segment
from typer.testing import CliRunner

from pyannote.audio import __main__ as cli


@pytest.mark.parametrize(
    "selected",
    [[], ["DiarizationCoverage"], ["DiarizationPurity", "DiarizationCoverage"]],
)
@pytest.mark.parametrize("per_file", [False, True])
def test_benchmark_configurations(tmp_path, monkeypatch, selected, per_file):
    reference = Annotation(uri="file")
    reference[Segment(0, 2)] = "speaker"
    files = [
        {"uri": "file", "annotation": reference, "annotated": reference.get_timeline()}
    ]
    loaded = []
    calls = []

    class Pipeline:
        def __init__(self, config):
            self.trial = config["params"]["trial"]
            loaded.append(self.trial)

        def to(self, device):
            return self

        def __call__(self, files):
            for file in files:
                calls.append(self.trial)
                prediction = (
                    reference.copy() if self.trial == 0 else Annotation(uri="file")
                )
                yield file, SimpleNamespace(speaker_diarization=prediction)

    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "pipeline": "mock",
                "optimization": {
                    "metrics": ["DiarizationPurity", "DiarizationCoverage"],
                    "status": {
                        "pareto_front": [
                            {
                                "trial": 0,
                                "params": {"trial": 0},
                                "values": {
                                    "DiarizationPurity": 1.0,
                                    "DiarizationCoverage": 0.5,
                                },
                            },
                            {
                                "trial": 1,
                                "params": {"trial": 1},
                                "values": {
                                    "DiarizationPurity": 0.5,
                                    "DiarizationCoverage": 1.0,
                                },
                            },
                        ]
                    },
                },
            }
        )
    )
    monkeypatch.setattr(
        cli.Pipeline, "from_pretrained", lambda config, **kw: Pipeline(config)
    )
    monkeypatch.setattr(
        cli.pyannote.database.registry,
        "get_protocol",
        lambda *a, **kw: SimpleNamespace(test=lambda: iter(files)),
    )
    monkeypatch.setattr(cli.Audio, "get_duration", lambda self, file: 2.0)
    result = CliRunner().invoke(
        cli.app,
        [
            "benchmark",
            str(pipeline),
            "Debug.SpeakerDiarization.Debug",
            str(tmp_path),
            "--device",
            "cpu",
            *[arg for name in selected for arg in ("--metric", name)],
            *(["--per-file"] if per_file else []),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    expected_trials = [
        0 if name == "DiarizationPurity" else 1
        for name in selected or ["DiarizationPurity"]
    ]
    assert loaded == calls == expected_trials
    for name, trial in zip(selected or ["DiarizationPurity"], expected_trials):
        stem = "Debug.SpeakerDiarization.Debug.test"
        output_dir = tmp_path / f"OptimizedFor{name}" if selected else tmp_path
        config = yaml.safe_load((output_dir / f"{stem}.config.yaml").read_text())
        assert config["params"]["trial"] == trial
        report = pd.read_csv(output_dir / f"{stem}.csv", header=[0, 1], index_col=0)
        assert report.loc["TOTAL", ("diarization error rate", "%")] == 100.0 * trial
        assert "diarization error rate" in (output_dir / f"{stem}.txt").read_text()
        assert (output_dir / f"{stem}.SpeakerCount.csv").exists()
        assert (output_dir / f"{stem}.yml").exists()
        rttm = (
            output_dir / stem / "rttm" / "file.rttm"
            if per_file
            else output_dir / f"{stem}.rttm"
        )
        assert rttm.exists()
    if selected:
        assert {path.name for path in tmp_path.iterdir()} == {
            "pipeline.yaml",
            *(f"OptimizedFor{name}" for name in selected),
        }
    assert not list(tmp_path.rglob("*OptimizedMinDurationOff*"))
