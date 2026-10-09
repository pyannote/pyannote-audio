# The MIT License (MIT)
#
# Copyright (c) 2024-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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

import matplotlib.pyplot as plt
import pytest
from lightning import Trainer
from pyannote.database import FileFinder, registry

from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.tasks import SpeakerDiarization, VoiceActivityDetection


def fit_and_get_figure(task, monkeypatch):
    # keep the figure that is logged every 2^n epochs
    figures = []
    monkeypatch.setattr(plt, "close", figures.append)

    model = SimpleSegmentationModel(task=task)
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)

    (figure,) = figures
    _, ncols = figure.axes[0].get_subplotspec().get_gridspec().get_geometry()

    # even rows contain the reference, odd rows the prediction
    return [i for i, ax in enumerate(figure.axes) if (i // ncols) % 2 == 0 and ax.lines]


@pytest.mark.parametrize("task_class", [SpeakerDiarization, VoiceActivityDetection])
@pytest.mark.parametrize("batch_size", [2, 5, 6])
def test_validation_figure_has_one_cell_per_sample(task_class, batch_size, monkeypatch):
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    task = task_class(protocol, duration=2.0, batch_size=batch_size, num_workers=0)
    assert len(fit_and_get_figure(task, monkeypatch)) == batch_size


def test_validation_figure_with_fewer_validation_chunks_than_batch_size(monkeypatch):
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    # with 20s chunks, the validation set is smaller than both batch_size and 9
    task = VoiceActivityDetection(protocol, duration=20.0, batch_size=32, num_workers=0)
    cells = fit_and_get_figure(task, monkeypatch)
    assert len(cells) == len(task.prepared_data["validation"])
