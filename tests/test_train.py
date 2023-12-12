from pathlib import Path

import pytest
from pyannote.database import FileFinder, get_protocol
from pytorch_lightning import Trainer

from pyannote.audio.models.embedding.debug import SimpleEmbeddingModel
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.tasks import (
    MultiLabelSegmentation,
    OverlappedSpeechDetection,
    SpeakerDiarization,
    SupervisedRepresentationLearningWithArcFace,
    VoiceActivityDetection,
)

CACHE_FILE_PATH = "./cache/cache_file"


@pytest.fixture()
def protocol():
    return get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


@pytest.fixture()
def gender_protocol():
    def to_gender(file):
        annotation = file["annotation"]
        mapping = {label: label[0] for label in annotation.labels()}
        return annotation.rename_labels(mapping)

    def classes(file):
        return ["M", "F"]

    return get_protocol(
        "Debug.SpeakerDiarization.Debug",
        preprocessors={
            "audio": FileFinder(),
            "annotation": to_gender,
            "classes": classes,
        },
    )


def test_train_segmentation(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_segmentation_with_cached_data_mono_device(protocol):
    first_task = SpeakerDiarization(protocol, cache=CACHE_FILE_PATH)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = SpeakerDiarization(protocol, cache=CACHE_FILE_PATH)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_train_multilabel_segmentation(gender_protocol):
    multilabel_segmentation = MultiLabelSegmentation(gender_protocol)
    model = SimpleSegmentationModel(task=multilabel_segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_multilabel_segmentation_with_cached_data_mono_device(gender_protocol):
    first_task = MultiLabelSegmentation(gender_protocol, cache=CACHE_FILE_PATH)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = MultiLabelSegmentation(gender_protocol, cache=CACHE_FILE_PATH)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_train_supervised_representation_with_arcface(protocol):
    supervised_representation_with_arface = SupervisedRepresentationLearningWithArcFace(
        protocol
    )
    model = SimpleEmbeddingModel(task=supervised_representation_with_arface)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_voice_activity_detection(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_voice_activity_detection_with_cached_data_mono_device(protocol):
    first_task = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_train_overlapped_speech_detection(protocol):
    overlapped_speech_detection = OverlappedSpeechDetection(protocol)
    model = SimpleSegmentationModel(task=overlapped_speech_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_overlapped_speech_detection_with_cached_data_mono_device(protocol):
    first_task = OverlappedSpeechDetection(protocol, cache=CACHE_FILE_PATH)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = OverlappedSpeechDetection(protocol, cache=CACHE_FILE_PATH)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_finetune_with_task_that_does_not_need_setup_for_specs(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_with_task_that_needs_setup_for_specs(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_with_task_that_needs_setup_for_specs_and_with_cache(protocol):
    segmentation = SpeakerDiarization(protocol, cache=CACHE_FILE_PATH)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol, cache=CACHE_FILE_PATH)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_transfer_with_task_that_does_not_need_setup_for_specs(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_transfer_with_task_that_needs_setup_for_specs(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_needs_setup_for_specs(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_needs_setup_for_specs_and_with_cache(protocol):
    segmentation = SpeakerDiarization(protocol, cache=CACHE_FILE_PATH)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs(protocol):
    vad = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol)
    model.task = vad
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs_and_with_cache(
    protocol,
):
    vad = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    model.task = vad
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs_and_with_cache(
    protocol,
):
    vad = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol, cache=CACHE_FILE_PATH)
    model.task = vad
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    Path(CACHE_FILE_PATH).unlink(missing_ok=True)


def test_transfer_freeze_with_task_that_does_not_need_setup_for_specs(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_transfer_freeze_with_task_that_needs_setup_for_specs(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    model.freeze_up_to("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)
