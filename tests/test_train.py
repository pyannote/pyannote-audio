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


def test_finetune_with_task_that_does_not_need_setup_for_specs(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_finetune_with_task_that_needs_setup_for_specs(protocol):
    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    segmentation = Segmentation(protocol)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_transfer_with_task_that_does_not_need_setup_for_specs(protocol):

    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_transfer_with_task_that_needs_setup_for_specs(protocol):

    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    segmentation = Segmentation(protocol)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_finetune_freeze_with_task_that_needs_setup_for_specs(protocol):

    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    segmentation = Segmentation(protocol)
    model.task = segmentation
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs(protocol):

    vad = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol)
    model.task = vad
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_transfer_freeze_with_task_that_does_not_need_setup_for_specs(protocol):

    segmentation = Segmentation(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def test_transfer_freeze_with_task_that_needs_setup_for_specs(protocol):

    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    segmentation = Segmentation(protocol)
    model.task = segmentation
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)
