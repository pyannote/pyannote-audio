import numpy as np
import pytest
import pytorch_lightning as pl

from pyannote.audio import Inference, Model
from pyannote.audio.core.task import Resolution
from pyannote.audio.models.segmentation.debug import (
    MultiTaskSegmentationModel,
    SimpleSegmentationModel,
)
from pyannote.audio.tasks import VoiceActivityDetection
from pyannote.audio.tasks.segmentation.multi_task_segmentation import (
    MultiTaskSegmentation,
)
from pyannote.core import SlidingWindowFeature
from pyannote.database import FileFinder, get_protocol

HF_SAMPLE_MODEL_ID = "pyannote/TestModelForContinuousIntegration"


def test_hf_download_inference():
    inference = Inference(HF_SAMPLE_MODEL_ID, device="cpu")
    assert isinstance(inference, Inference)


def test_hf_download_model():
    model = Model.from_pretrained(HF_SAMPLE_MODEL_ID)
    assert isinstance(model, Model)


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


@pytest.fixture()
def pretrained_model():
    return Model.from_pretrained(HF_SAMPLE_MODEL_ID)


@pytest.fixture()
def dev_file():
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    return next(protocol.development())


def test_duration_warning(trained):
    protocol, model = trained
    with pytest.warns(UserWarning):
        duration = model.specifications.duration
        new_duration = duration + 1
        Inference(model, duration=new_duration, step=0.1, batch_size=128)


def test_step_check_warning(trained):
    protocol, model = trained
    with pytest.raises(ValueError):
        duration = model.specifications.duration
        Inference(model, step=duration + 1, batch_size=128)


def test_invalid_window_fails(trained):
    protocol, model = trained
    with pytest.raises(ValueError):
        Inference(model, window="unknown")


def test_invalid_resolution_fails(trained):
    protocol, model = trained
    with pytest.warns(UserWarning):
        model.specifications.resolution = Resolution.FRAME
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


def test_skip_aggregation(pretrained_model, dev_file):

    inference = Inference(pretrained_model, skip_aggregation=True)
    scores = inference(dev_file)
    assert len(scores.data.shape) == 3
