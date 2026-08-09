# MIT License
#
# Copyright (c) 2025- pyannoteAI
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

"""Tests for the SpeakerDiarization pipeline"""

import pytest
from pyannote.database import FileFinder, registry

from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import SpeakerDiarization, speaker_diarization
from pyannote.audio.tasks import SpeakerDiarization as SpeakerDiarizationTask


@pytest.fixture(scope="module")
def segmentation_model():
    """Locally instantiated model, so that no test below needs a download."""
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    task = SpeakerDiarizationTask(
        protocol, duration=2.0, batch_size=2, num_workers=0, max_speakers_per_chunk=3
    )
    model = SimpleSegmentationModel(task=task)
    model.prepare_data()
    model.setup()
    return model


@pytest.mark.parametrize(
    "clustering",
    ["AgglomerativeClustering", "KMeansClustering", "OracleClustering"],
)
def test_plda_not_loaded_when_clustering_does_not_use_it(
    segmentation_model, monkeypatch, clustering
):
    """Only VBx clustering consumes the PLDA, so only it should load one.

    The default PLDA lives in the gated `pyannote/speaker-diarization-community-1`
    repo: loading it for every clustering method makes `speaker-diarization-3.1`
    (which pins AgglomerativeClustering) fail for users who only accepted the
    repos its model card lists.
    """

    def unexpected_get_plda(*args, **kwargs):
        raise AssertionError(f"PLDA should not be loaded for {clustering}")

    monkeypatch.setattr(speaker_diarization, "get_plda", unexpected_get_plda)

    pipeline = SpeakerDiarization(
        segmentation=segmentation_model,
        embedding=segmentation_model,
        clustering=clustering,
    )
    assert pipeline._plda is None


def test_plda_loaded_for_vbx_clustering(segmentation_model, monkeypatch):
    """VBx clustering still gets its PLDA."""

    plda = object()
    monkeypatch.setattr(speaker_diarization, "get_plda", lambda *args, **kwargs: plda)

    pipeline = SpeakerDiarization(
        segmentation=segmentation_model,
        embedding=segmentation_model,
        clustering="VBxClustering",
    )
    assert pipeline._plda is plda
    assert pipeline.clustering.plda is plda
