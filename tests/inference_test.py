import numpy as np
import pytest
import torch
from lightning import Trainer
from pyannote.core import SlidingWindowFeature
from pyannote.database import FileFinder, registry

from pyannote.audio import Audio, Inference, Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import (
    VoiceActivityDetection as VoiceActivityDetectionPipeline,
)
from pyannote.audio.tasks import VoiceActivityDetection

HF_SAMPLE_MODEL_ID = "pyannote/ci-segmentation"


@pytest.mark.network
def test_hf_download_inference():
    model = Model.from_pretrained(HF_SAMPLE_MODEL_ID)
    inference = Inference(model, device="cpu")
    assert isinstance(inference, Inference)


@pytest.mark.network
def test_hf_download_model():
    model = Model.from_pretrained(HF_SAMPLE_MODEL_ID)
    assert isinstance(model, Model)


@pytest.fixture()
def trained():
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    vad = VoiceActivityDetection(protocol, duration=2.0, batch_size=16, num_workers=4)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)
    return protocol, model


@pytest.fixture()
def pretrained_model():
    return Model.from_pretrained(HF_SAMPLE_MODEL_ID)


@pytest.fixture()
def dev_file():
    protocol = registry.get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    return next(protocol.development())


def test_duration_warning(trained):
    _protocol, model = trained
    with pytest.warns(UserWarning):
        duration = model.specifications.duration
        new_duration = duration + 1
        Inference(model, duration=new_duration, step=0.1, batch_size=128)


def test_step_check_warning(trained):
    _protocol, model = trained
    with pytest.raises(ValueError):
        duration = model.specifications.duration
        Inference(model, step=duration + 1, batch_size=128)


def test_invalid_window_fails(trained):
    _protocol, model = trained
    with pytest.raises(ValueError):
        Inference(model, window="unknown")


def test_invalid_resolution_fails(trained):
    _protocol, model = trained
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
    _protocol, model = trained
    inference = Inference(model, batch_size=128)
    output = inference("tests/data/dev00.wav")
    assert isinstance(output, SlidingWindowFeature)


@pytest.mark.network
def test_skip_aggregation(pretrained_model, dev_file):
    inference = Inference(pretrained_model, skip_aggregation=True)
    scores = inference(dev_file)
    assert len(scores.data.shape) == 3


class AlwaysSpeech(PyanNet):
    def forward(self, waveforms):
        return torch.ones_like(super().forward(waveforms))


# 2.2s lies on the chunk grid (2s chunks, 0.2s step), others need padding
@pytest.mark.parametrize("duration", [0.5, 2.2, 3.3, 4.7])
def test_speech_does_not_extend_past_end_of_file(duration):
    model = AlwaysSpeech()
    model.specifications = Specifications(
        problem=Problem.BINARY_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=2.0,
        classes=["speech"],
    )
    model.build()

    pipeline = VoiceActivityDetectionPipeline(segmentation=model)
    pipeline.instantiate(
        {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.0, "min_duration_off": 0.0}
    )

    sample_rate = 16000
    file = {
        "waveform": torch.zeros(1, int(duration * sample_rate)),
        "sample_rate": sample_rate,
    }
    last_region = pipeline(file).get_timeline()[-1]
    assert last_region.end <= duration
    # cutting the audio along the detected speech must not raise
    Audio().crop(file, last_region)
