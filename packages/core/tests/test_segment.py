from pyannote.core import Segment
from tests.utils import preserve_segment_state


def test_creation():

    s = 1.0
    e = 9.0
    segment = Segment(start=s, end=e)

    assert str(segment) == "[ 00:00:01.000 -->  00:00:09.000]"
    assert segment.end == 9
    assert segment.duration == 8
    assert segment.middle == 5


def test_intersection():

    segment = Segment(start=1, end=9)
    other_segment = Segment(4, 13)

    assert segment.intersects(other_segment)
    assert segment & other_segment == Segment(4, 9)

    other_segment = Segment(13, 20)

    assert not segment.intersects(other_segment)


def test_inclusion():

    segment = Segment(start=1, end=9)
    other_segment = Segment(5, 9)

    print(other_segment in segment)
    assert other_segment in segment
    assert not segment.overlaps(23)


def test_other_operation():

    segment = Segment(start=1, end=9)
    other_segment = Segment(10, 30)
    assert segment | other_segment == Segment(1, 30)

    other_segment = Segment(14, 15)
    assert segment ^ other_segment == Segment(9, 14)


@preserve_segment_state
def test_segment_precision_mode():
    Segment.set_precision(None)
    assert Segment(90 / 1000, 90 / 1000 + 240 / 1000) != Segment(90 / 1000, 330 / 1000)
    Segment.set_precision(4)
    assert Segment(90 / 1000, 90 / 1000 + 240 / 1000) == Segment(90 / 1000, 330 / 1000)
