# MIT License
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

import pytest
import torch

from pyannote.audio.torchmetrics.functional.audio.diarization_error_rate import (
    _der_update,
    diarization_error_rate,
)


@pytest.fixture
def target():
    chunk1 = [[0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 1]]
    chunk2 = [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0]]
    return torch.tensor([chunk1, chunk2], dtype=torch.float32).transpose(2, 1)


@pytest.fixture
def prediction():
    chunk1 = [[0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1], [1, 0]]
    chunk2 = [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 1], [1, 0]]
    return torch.tensor([chunk1, chunk2], dtype=torch.float32).transpose(2, 1)


def test_frame_reduction(target, prediction):
    false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
        prediction, target, reduce="frame"
    )

    torch.testing.assert_close(
        false_alarm,
        torch.Tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
        ),
    )

    torch.testing.assert_close(
        missed_detection,
        torch.Tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    torch.testing.assert_close(
        speaker_confusion,
        torch.Tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
    )

    torch.testing.assert_close(
        speech_total,
        torch.Tensor(
            [[0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
        ),
    )


def test_chunk_reduction(target, prediction):
    false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
        prediction, target, reduce="chunk"
    )

    torch.testing.assert_close(
        false_alarm,
        torch.Tensor([1.0, 2.0]),
    )

    torch.testing.assert_close(
        missed_detection,
        torch.Tensor([2.0, 0.0]),
    )

    torch.testing.assert_close(
        speaker_confusion,
        torch.Tensor([1.0, 0.0]),
    )

    torch.testing.assert_close(
        speech_total,
        torch.Tensor([8.0, 4.0]),
    )


def test_batch_reduction(target, prediction):
    false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
        prediction, target, reduce="batch"
    )
    torch.testing.assert_close(false_alarm.item(), 3.0)
    torch.testing.assert_close(missed_detection.item(), 2.0)
    torch.testing.assert_close(speaker_confusion.item(), 1.0)
    torch.testing.assert_close(speech_total.item(), 12.0)


def test_batch_der(target, prediction):
    der = diarization_error_rate(prediction, target, reduce="batch")
    torch.testing.assert_close(der.item(), (3.0 + 2.0 + 1.0) / 12.0)


def test_batch_der_with_components(target, prediction):
    der, (
        false_alarm,
        missed_detection,
        speaker_confusion,
        speech_total,
    ) = diarization_error_rate(
        prediction, target, reduce="batch", return_components=True
    )
    torch.testing.assert_close(der.item(), (3.0 + 2.0 + 1.0) / 12.0)
    torch.testing.assert_close(false_alarm.item(), 3.0)
    torch.testing.assert_close(missed_detection.item(), 2.0)
    torch.testing.assert_close(speaker_confusion.item(), 1.0)
    torch.testing.assert_close(speech_total.item(), 12.0)


def test_chunk_der(target, prediction):
    der = diarization_error_rate(prediction, target, reduce="chunk")
    torch.testing.assert_close(der, torch.Tensor([4.0 / 8.0, 2.0 / 4.0]))
