from pyannote.core import Segment, Timeline

from pyannote.metrics.segmentation import SegmentationPrecision


def test_segmentation_precision_empty_hypothesis():
    ref = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    # empty hypothesis with >=2 reference segments used to crash np.zeros((N, -1))
    assert SegmentationPrecision()(ref, Timeline()) == 1.0

    # accumulating an empty hypothesis used to push a boundary count of -1,
    # giving a corpus precision > 1
    m = SegmentationPrecision()
    m(ref, ref)
    m(Timeline([Segment(0, 3)]), Timeline())
    assert abs(m) <= 1.0
