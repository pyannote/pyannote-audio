import pytest

from pyannote.core import Segment

def test_dummy():
    assert isinstance(Segment(1., 2.), Segment)
