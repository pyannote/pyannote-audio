import numpy.testing as npt
import pytest
from pyannote.core import Annotation, Segment

from pyannote.metrics.identification import (
    IdentificationErrorRate,
    IdentificationPrecision,
    IdentificationRecall,
)


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "A"
    hypothesis[Segment(13, 14)] = "D"
    hypothesis[Segment(14, 20)] = "B"
    hypothesis[Segment(22, 38)] = "C"
    hypothesis[Segment(38, 40)] = "D"
    return hypothesis


def test_error_rate(reference, hypothesis):
    identificationErrorRate = IdentificationErrorRate()
    error_rate = identificationErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.5161290322580645, decimal=7)


def test_detailed(reference, hypothesis):
    identificationErrorRate = IdentificationErrorRate()
    details = identificationErrorRate(reference, hypothesis, detailed=True)

    confusion = details["confusion"]
    npt.assert_almost_equal(confusion, 7.0, decimal=7)

    correct = details["correct"]
    npt.assert_almost_equal(correct, 22.0, decimal=7)

    rate = details["identification error rate"]
    npt.assert_almost_equal(rate, 0.5161290322580645, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 7.0, decimal=7)

    missed_detection = details["missed detection"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 31.0, decimal=7)


def test_precision(reference, hypothesis):
    identificationPrecisions = IdentificationPrecision()
    precision = identificationPrecisions(reference, hypothesis)
    npt.assert_almost_equal(precision, 0.611, decimal=3)


def test_recall(reference, hypothesis):
    identificationRecall = IdentificationRecall()
    recall = identificationRecall(reference, hypothesis)
    npt.assert_almost_equal(recall, 0.710, decimal=3)
