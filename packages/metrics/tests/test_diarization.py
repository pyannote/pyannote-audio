import numpy.testing as npt
import pytest
from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.diarization import (
    DiarizationCoverage,
    DiarizationErrorRate,
    DiarizationPurity,
    DiarizationSpeakerCountAccuracy,
    DiarizationSpeakerCountError,
)


def speaker_annotation(count, duration=1.0):
    annotation = Annotation()
    for speaker in range(count):
        annotation[Segment(0, duration), speaker] = f"speaker_{speaker}"
        annotation[Segment(duration, 2 * duration), speaker] = f"speaker_{speaker}"
    return annotation


@pytest.mark.parametrize(
    "expected,predicted", [(3, 3), (3, 1), (1, 4), (0, 0), (0, 3), (3, 0)]
)
def test_speaker_count(expected, predicted):
    reference = speaker_annotation(expected)
    hypothesis = speaker_annotation(predicted).rename_labels(generator="int")
    uem = Timeline([Segment(0, 2)])
    assert DiarizationSpeakerCountAccuracy()(reference, hypothesis, uem=uem) == float(
        expected == predicted
    )
    assert DiarizationSpeakerCountError()(reference, hypothesis, uem=uem) == abs(
        predicted - expected
    )


@pytest.mark.parametrize(
    "metric_class,component,scores",
    [
        (DiarizationSpeakerCountAccuracy, "correct", [1.0, 0.0, 0.0]),
        (DiarizationSpeakerCountError, "error", [0.0, 2.0, 3.0]),
    ],
)
def test_speaker_count_accumulation(metric_class, component, scores):
    metric = metric_class()
    assert abs(metric) == 0.0
    for index, (expected, predicted, duration) in enumerate(
        [(3, 3, 1), (3, 1, 10), (1, 4, 100)]
    ):
        details = metric(
            speaker_annotation(expected, duration),
            speaker_annotation(predicted, duration),
            uem=Timeline([Segment(0, 2 * duration)]),
            uri=f"file_{index}",
            detailed=True,
        )
        assert details == {
            component: scores[index],
            "files": 1.0,
            metric.name: scores[index],
        }
    assert metric["files"] == 3
    assert metric[component] == sum(scores)
    assert abs(metric) == pytest.approx(sum(scores) / 3)
    report = metric.report()
    unit = "" if metric_class is DiarizationSpeakerCountError else "%"
    scale = 1 if unit == "" else 100
    assert report.loc["TOTAL", (metric.name, unit)] == pytest.approx(
        scale * sum(scores) / 3
    )
    assert report.loc["file_1", (metric.name, unit)] == scale * scores[1]
    metric.reset()
    assert metric[:] == {component: 0.0, "files": 0.0}
    assert abs(metric) == 0.0


@pytest.mark.parametrize(
    "metric_class", [DiarizationSpeakerCountAccuracy, DiarizationSpeakerCountError]
)
def test_speaker_count_uem(metric_class):
    reference = speaker_annotation(2)
    hypothesis = speaker_annotation(1)
    reference[Segment(10, 20)] = "outside_reference"
    hypothesis[Segment(10, 20)] = "outside_hypothesis_1"
    hypothesis[Segment(30, 40)] = "outside_hypothesis_2"
    metric = metric_class()
    accuracy = metric_class is DiarizationSpeakerCountAccuracy
    with pytest.warns(UserWarning, match="uem"):
        assert metric(reference, hypothesis) == (1.0 if accuracy else 0.0)
    assert metric(reference, hypothesis, uem=Timeline([Segment(0, 2)])) == (
        0.0 if accuracy else 1.0
    )
    assert metric(reference, hypothesis, uem=Timeline()) == (1.0 if accuracy else 0.0)
    assert len(reference.labels()) == 3
    assert len(hypothesis.labels()) == 3


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def reference_with_overlap():
    reference = Annotation()
    reference[Segment(0, 13)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "a"
    hypothesis[Segment(13, 14)] = "d"
    hypothesis[Segment(14, 20)] = "b"
    hypothesis[Segment(22, 38)] = "c"
    hypothesis[Segment(38, 40)] = "d"
    return hypothesis


def test_error_rate(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    error_rate = diarizationErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.5161290322580645, decimal=7)


def test_optimal_mapping(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    mapping = diarizationErrorRate.optimal_mapping(reference, hypothesis)
    assert mapping == {"a": "A", "b": "B", "c": "C"}


def test_detailed(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    details = diarizationErrorRate(reference, hypothesis, detailed=True)

    confusion = details["confusion"]
    npt.assert_almost_equal(confusion, 7.0, decimal=7)

    correct = details["correct"]
    npt.assert_almost_equal(correct, 22.0, decimal=7)

    rate = details["diarization error rate"]
    npt.assert_almost_equal(rate, 0.5161290322580645, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 7.0, decimal=7)

    missed_detection = details["missed detection"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 31.0, decimal=7)


def test_purity(reference, hypothesis):
    diarizationPurity = DiarizationPurity()
    purity = diarizationPurity(reference, hypothesis)
    npt.assert_almost_equal(purity, 0.6666, decimal=3)


def test_coverage(reference, hypothesis):
    diarizationCoverage = DiarizationCoverage()
    coverage = diarizationCoverage(reference, hypothesis)
    npt.assert_almost_equal(coverage, 0.7096, decimal=3)


def test_skip_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=True)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 32, decimal=3)


def test_leep_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=False)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 34, decimal=3)


def test_bug_16():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    hypothesis = Annotation()

    metric = DiarizationErrorRate(collar=1)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 9, decimal=3)

    metric = DiarizationErrorRate(collar=0)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 10, decimal=3)


def test_jaccard_error_rate_empty_reference():
    from pyannote.metrics.diarization import JaccardErrorRate

    hypothesis = Annotation()
    hypothesis[Segment(0, 10)] = "spk"
    # empty reference -> zero speaker count -> must not ZeroDivisionError
    assert JaccardErrorRate()(Annotation(), hypothesis) == 1.0
