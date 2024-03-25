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


from tempfile import mkstemp

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


@pytest.fixture()
def protocol():
    return get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )


@pytest.fixture()
def cache():
    return mkstemp()[1]


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


def test_train_segmentation_with_cached_data_mono_device(protocol, cache):
    first_task = SpeakerDiarization(protocol, cache=cache)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = SpeakerDiarization(protocol, cache=cache)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)


def test_train_multilabel_segmentation(gender_protocol):
    multilabel_segmentation = MultiLabelSegmentation(gender_protocol)
    model = SimpleSegmentationModel(task=multilabel_segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_multilabel_segmentation_with_cached_data_mono_device(
    gender_protocol, cache
):
    first_task = MultiLabelSegmentation(gender_protocol, cache=cache)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = MultiLabelSegmentation(gender_protocol, cache=cache)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)


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


def test_train_voice_activity_detection_with_cached_data_mono_device(protocol, cache):
    first_task = VoiceActivityDetection(protocol, cache=cache)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = VoiceActivityDetection(protocol, cache=cache)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)


def test_train_overlapped_speech_detection(protocol):
    overlapped_speech_detection = OverlappedSpeechDetection(protocol)
    model = SimpleSegmentationModel(task=overlapped_speech_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_train_overlapped_speech_detection_with_cached_data_mono_device(
    protocol, cache
):
    first_task = OverlappedSpeechDetection(protocol, cache=cache)
    first_model = SimpleSegmentationModel(task=first_task)
    first_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    first_trainer.fit(first_model)

    second_task = OverlappedSpeechDetection(protocol, cache=cache)
    second_model = SimpleSegmentationModel(task=second_task)
    second_trainer = Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    second_trainer.fit(second_model)


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


def test_finetune_with_task_that_needs_setup_for_specs_and_with_cache(protocol, cache):
    segmentation = SpeakerDiarization(protocol, cache=cache)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol, cache=cache)
    model.task = segmentation
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


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
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_needs_setup_for_specs_and_with_cache(
    protocol, cache
):
    segmentation = SpeakerDiarization(protocol, cache=cache)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol, cache=cache)
    model.task = segmentation
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs(protocol):
    vad = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol)
    model.task = vad
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_finetune_freeze_with_task_that_does_not_need_setup_for_specs_and_with_cache(
    protocol,
    cache,
):
    vad = VoiceActivityDetection(protocol, cache=cache)
    model = SimpleSegmentationModel(task=vad)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    vad = VoiceActivityDetection(protocol, cache=cache)
    model.task = vad
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_transfer_freeze_with_task_that_does_not_need_setup_for_specs(protocol):
    segmentation = SpeakerDiarization(protocol)
    model = SimpleSegmentationModel(task=segmentation)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    voice_activity_detection = VoiceActivityDetection(protocol)
    model.task = voice_activity_detection
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)


def test_transfer_freeze_with_task_that_needs_setup_for_specs(protocol):
    voice_activity_detection = VoiceActivityDetection(protocol)
    model = SimpleSegmentationModel(task=voice_activity_detection)
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)

    segmentation = SpeakerDiarization(protocol)
    model.task = segmentation
    model.freeze_by_name("mfcc")
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model)
