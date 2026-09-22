# MIT License
#
# Copyright (c) 2026- pyannoteAI
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
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from pyannote.audio.pipelines.utils.diarization import SpeakerDiarizationMixin


@pytest.fixture
def reference():
    annotation = Annotation()
    annotation[Segment(0, 10)] = "A"
    annotation[Segment(12, 20)] = "B"
    annotation[Segment(24, 27)] = "A"
    annotation[Segment(30, 40)] = "C"
    return annotation


@pytest.fixture
def hypothesis():
    annotation = Annotation()
    annotation[Segment(2, 13)] = "a"
    annotation[Segment(13, 14)] = "d"
    annotation[Segment(14, 20)] = "b"
    annotation[Segment(22, 38)] = "c"
    annotation[Segment(38, 40)] = "d"
    return annotation


@pytest.fixture
def annotated():
    return Timeline([Segment(10, 20)])


def test_optimal_mapping_accepts_mapping(reference, hypothesis):
    _, mapping = SpeakerDiarizationMixin.optimal_mapping(
        {"annotation": reference}, hypothesis, return_mapping=True
    )
    assert mapping == {"a": "A", "b": "B", "c": "C"}


def test_optimal_mapping_uses_annotated_as_uem(reference, hypothesis, annotated):
    _, mapping = SpeakerDiarizationMixin.optimal_mapping(
        {"annotation": reference, "annotated": annotated},
        hypothesis,
        return_mapping=True,
    )

    expected = DiarizationErrorRate().optimal_mapping(
        reference, hypothesis, uem=annotated
    )
    assert mapping == expected

    # the UEM is not cosmetic: it changes which labels can be mapped
    assert mapping != DiarizationErrorRate().optimal_mapping(reference, hypothesis)
