# The MIT License (MIT)

# Copyright (c) 2012-2017 CNRS

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

from pyannote.core import Annotation, Timeline


class SegmentationErrorAnalysis:
    def __init__(self):
        super().__init__()

    def __call__(
        self, reference: Timeline | Annotation, hypothesis: Timeline | Annotation
    ) -> Annotation:

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        # over-segmentation
        over = Timeline(uri=reference.uri)
        prev_r = reference[0]
        intersection = []
        for r, h in reference.co_iter(hypothesis):
            if r != prev_r:
                intersection = sorted(intersection)
                for _, segment in intersection[:-1]:
                    over.add(segment)
                intersection = []
                prev_r = r

            segment = r & h
            intersection.append((segment.duration, segment))

        intersection = sorted(intersection)
        for _, segment in intersection[:-1]:
            over.add(segment)

        # under-segmentation
        under = Timeline(uri=reference.uri)
        prev_h = hypothesis[0]
        intersection = []
        for h, r in hypothesis.co_iter(reference):
            if h != prev_h:
                intersection = sorted(intersection)
                for _, segment in intersection[:-1]:
                    under.add(segment)
                intersection = []
                prev_h = h

            segment = h & r
            intersection.append((segment.duration, segment))

        intersection = sorted(intersection)
        for _, segment in intersection[:-1]:
            under.add(segment)

        # extent
        extent = reference.extent()

        # frontier error (both under- and over-segmented)
        frontier = under.crop(over)

        # under-segmented
        not_over = over.gaps(support=extent)
        only_under = under.crop(not_over)

        # over-segmented
        not_under = under.gaps(support=extent)
        only_over = over.crop(not_under)

        status = Annotation(uri=reference.uri)
        for segment in frontier:
            status[segment, "_"] = "shift"
        for segment in only_over:
            status[segment, "_"] = "over-segmentation"
        for segment in only_under:
            status[segment, "_"] = "under-segmentation"

        return status.support()
