import pytest
from pyannote.core import Annotation, Segment

from pyannote.metrics.segmentation import (
    SegmentationCoverage,
    SegmentationPurity,
    SegmentationPurityCoverageFMeasure,
)

METRICS = [
    SegmentationCoverage,
    SegmentationPurity,
    SegmentationPurityCoverageFMeasure,
]


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 20)] = "A"
    reference[Segment(20, 40)] = "B"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(0, 5)] = "a"
    hypothesis[Segment(5, 25)] = "b"
    hypothesis[Segment(25, 40)] = "c"
    return hypothesis


@pytest.mark.parametrize("metric_class", METRICS)
def test_empty_annotations(metric_class):
    """Empty annotations produce an empty cooccurrence matrix, which used to
    raise ValueError from np.max (and ZeroDivisionError once reached)."""
    metric = metric_class()
    assert metric(Annotation(), Annotation()) == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_empty_hypothesis(metric_class, reference):
    metric = metric_class()
    assert metric(reference, Annotation()) == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_empty_reference(metric_class, hypothesis):
    metric = metric_class()
    assert metric(Annotation(), hypothesis) == 1.0


def test_perfect_segmentation(reference):
    for metric_class in METRICS:
        assert metric_class()(reference, reference) == 1.0


def test_imperfect_segmentation(reference, hypothesis):
    """Guarding the empty case must not change the values for real input."""
    assert SegmentationPurity()(reference, hypothesis) == pytest.approx(0.875)
    assert SegmentationCoverage()(reference, hypothesis) == pytest.approx(0.75)
    assert SegmentationPurityCoverageFMeasure()(reference, hypothesis) == pytest.approx(
        0.8076923076923077
    )
