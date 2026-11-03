# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import numpy.testing as npt
import pytest
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.detection import (
    DetectionAccuracy,
    DetectionCostFunction,
    DetectionErrorRate,
    DetectionPrecision,
    DetectionPrecisionRecallFMeasure,
    DetectionRecall,
)

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Reference   |--------------|  |-----------|     |-----|  |--------------|

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Hypothesis     |-----------------|-----|     |-----------------|  |-----|
#                                  |--------|

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# UEM            |--------------------------------------|


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 5)] = "A"
    reference[Segment(6, 10)] = "B"
    reference[Segment(12, 14)] = "A"
    reference[Segment(15, 20)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(1, 7)] = "A"
    hypothesis[Segment(7, 9)] = "D"
    hypothesis[Segment(7, 10)] = "B"
    hypothesis[Segment(11, 17)] = "C"
    hypothesis[Segment(18, 20)] = "D"
    return hypothesis


@pytest.fixture
def uem():
    return Timeline([Segment(1, 14)])


def test_error_rate(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    error_rate = detectionErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.3125, decimal=7)


def test_detailed(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    details = detectionErrorRate(reference, hypothesis, detailed=True)

    rate = details["detection error rate"]
    npt.assert_almost_equal(rate, 0.3125, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 3.0, decimal=7)

    missed_detection = details["miss"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 16.0, decimal=7)


def test_accuracy(reference, hypothesis):
    # 15 correct / 20 total
    detectionAccuracy = DetectionAccuracy()
    accuracy = detectionAccuracy(reference, hypothesis)
    npt.assert_almost_equal(accuracy, 0.75, decimal=3)


def test_precision(reference, hypothesis):
    # 14 true positive / 17 detected
    detectionPrecision = DetectionPrecision()
    precision = detectionPrecision(reference, hypothesis)
    npt.assert_almost_equal(precision, 0.8235, decimal=3)


def test_recall(reference, hypothesis):
    # 14 true positive / 16 expected
    detectionRecall = DetectionRecall()
    recall = detectionRecall(reference, hypothesis)
    npt.assert_almost_equal(recall, 0.875, decimal=3)


def test_fscore(reference, hypothesis):
    # expected 28/33 since it
    # is computed as :
    # 2*precision*recall / (precision+recall)
    detectionFMeasure = DetectionPrecisionRecallFMeasure()

    fscore = detectionFMeasure(reference, hypothesis)
    npt.assert_almost_equal(fscore, 0.848, decimal=3)


def test_decision_cost_function(reference, hypothesis, uem):
    # No UEM.
    expected = 0.28125
    dcf = DetectionCostFunction(fa_weight=0.25, miss_weight=0.75)
    actual = dcf(reference, hypothesis)
    npt.assert_almost_equal(actual, expected, decimal=7)

    # UEM.
    expected = 1 / 6.0
    dcf = DetectionCostFunction(fa_weight=0.25, miss_weight=0.75)
    actual = dcf(reference, hypothesis, uem=uem)
    npt.assert_almost_equal(actual, expected, decimal=7)
