# The MIT License (MIT)
#
# Copyright (c) 2024- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import subprocess

import pytest
from pyannote.database import FileFinder, get_protocol


@pytest.fixture()
def protocol():
    return get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


@pytest.fixture()
def database():
    return "./tests/data/database.yml"


@pytest.fixture()
def model():
    return "pyannote/ci-segmentation"


def test_cli_train_vad(database, protocol):
    res = subprocess.run(
        [
            "pyannote-audio-train",
            "model=DebugSegmentation",
            "task=VoiceActivityDetection",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "trainer=fast_dev_run",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0


def test_cli_train_segmentation(database, protocol):
    res = subprocess.run(
        [
            "pyannote-audio-train",
            "model=DebugSegmentation",
            "task=SpeakerDiarization",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "trainer=fast_dev_run",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0


def test_cli_train_osd(database, protocol):
    res = subprocess.run(
        [
            "pyannote-audio-train",
            "model=DebugSegmentation",
            "task=OverlappedSpeechDetection",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "trainer=fast_dev_run",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0


def test_cli_train_supervised_representation_with_arcface(database, protocol):
    res = subprocess.run(
        [
            "pyannote-audio-train",
            "model=DebugEmbedding",
            "task=SpeakerEmbedding",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "trainer=fast_dev_run",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0


def test_cli_train_segmentation_with_pyannet(database, protocol):
    res = subprocess.run(
        [
            "pyannote-audio-train",
            "model=PyanNet",
            "task=SpeakerDiarization",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "trainer=fast_dev_run",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0


def test_cli_eval_segmentation_model(database, protocol, model):
    res = subprocess.run(
        [
            "pyannote-audio-eval",
            f"model={model}",
            f"+registry={database}",
            f"protocol={protocol.name}",
            "hydra.run.dir=.",  # run hydra app in current directory
            "hydra.output_subdir=null",  # disable hydra outputs
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    assert res.returncode == 0
