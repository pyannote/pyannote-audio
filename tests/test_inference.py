import numpy as np
import pytest
import pytorch_lightning as pl

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.task import Scale
from pyannote.audio.models.segmentation.debug import (
    MultiTaskSegmentationModel,
    SimpleSegmentationModel,
)
from pyannote.audio.tasks import MultiTaskSegmentation, VoiceActivityDetection
from pyannote.core import SlidingWindowFeature
from pyannote.database import FileFinder, get_protocol


@pytest.fixture()
def trained():
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    vad = VoiceActivityDetection(protocol, duration=2.0, batch_size=16, num_workers=4)
    model = SimpleSegmentationModel(task=vad)
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, vad)
    return protocol, model


def test_duration_warning(trained):
    protocol, model = trained
    with pytest.warns(UserWarning):
        duration = model.hparams.task_specifications.duration
        new_duration = duration + 1
        Inference(model, duration=new_duration, step=0.1, batch_size=128)


def test_step_check_warning(trained):
    protocol, model = trained
    with pytest.raises(ValueError):
        duration = model.hparams.task_specifications.duration
        Inference(model, step=duration + 1, batch_size=128)


def test_invalid_window_fails(trained):
    protocol, model = trained
    with pytest.raises(ValueError):
        Inference(model, window="unknown")


def test_invalid_scale_fails(trained):
    protocol, model = trained
    with pytest.warns(UserWarning):
        model.hparams.task_specifications.scale = Scale.FRAME
        Inference(model, window="whole", batch_size=128)


def test_whole_window_slide(trained):
    protocol, model = trained
    inference = Inference(model, window="whole", batch_size=128)
    dev_file = next(protocol.development())
    output = inference(dev_file)
    assert isinstance(output, np.ndarray)


def test_on_file_path(trained):
    protocol, model = trained
    inference = Inference(model, batch_size=128)
    output = inference("tests/data/dev00.wav")
    assert isinstance(output, SlidingWindowFeature)


def test_multi_seg_infer():
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    xseg = MultiTaskSegmentation(
        protocol,
        duration=2.0,
        vad=True,
        scd=True,
        osd=True,
        batch_size=32,
        num_workers=4,
    )
    model = MultiTaskSegmentationModel(task=xseg)
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)
    _ = trainer.fit(model, xseg)
    inference = Inference(model, duration=2.0, step=0.5)
    dev_file = next(protocol.development())
    scores = inference(dev_file)

    assert isinstance(scores, dict)

    for attr in ["vad", "scd", "osd"]:
        assert attr in scores
        assert isinstance(scores[attr], SlidingWindowFeature)
