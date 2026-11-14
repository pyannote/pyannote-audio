import numpy as np
import pytest
from pyannote.core import SlidingWindow, SlidingWindowFeature

from pyannote.audio.utils.signal import Binarize

# frame middles are at 0.01, 0.03, 0.05, ...
FRAMES = SlidingWindow(start=0.0, duration=0.02, step=0.02)


def test_binarize_no_frames():
    scores = SlidingWindowFeature(np.zeros((0, 1)), FRAMES)
    assert len(Binarize()(scores)) == 0


def test_binarize_single_active_frame():
    scores = SlidingWindowFeature(np.array([[0.9]]), FRAMES)
    (segment,) = Binarize(pad_offset=0.1)(scores).itersegments()
    assert (segment.start, segment.end) == pytest.approx((0.01, 0.11))


def test_binarize_active_until_last_frame():
    scores = SlidingWindowFeature(np.array([[0.1], [0.9], [0.9]]), FRAMES)
    (segment,) = Binarize()(scores).itersegments()
    assert (segment.start, segment.end) == pytest.approx((0.03, 0.05))
