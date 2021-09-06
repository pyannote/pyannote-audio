import pytest
from pytorch_lightning import Trainer

from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.tasks import (
    OverlappedSpeechDetection,
    Segmentation,
    VoiceActivityDetection,
)
from pyannote.database import FileFinder, get_protocol


@pytest.fixture()
def protocol():
    return get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


def test_train_segmentation(protocol):
    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_train_voice_activity_detection(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_train_overlapped_speech_detection(protocol):
    overlapped_speech_detection = OverlappedSpeechDetection(protocol)
    model = SimpleSegmentationModel(task=overlapped_speech_detection)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_finetune(protocol):
    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    segmentation = Segmentation(protocol)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_transfer(protocol):
    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
