#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2022- CNRS

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

from pyannote.database import ProtocolFile
from pyannote.core import Annotation, Segment


class LowerTemporalResolution:
    """Artificially degrade temporal resolution of reference annotation
    
    Parameters
    ----------
    resolution : float, optional
        Target temporal resolution, in seconds. Defaults to 0.1 (100ms).
    """

    preprocessed_key = "annotation"

    def __init__(self, resolution: float = 0.1):
        super().__init__()
        self.resolution = resolution

    def __call__(self, current_file: ProtocolFile) -> Annotation:

        annotation = current_file["annotation"]
        new_annotation = annotation.empty()

        for new_track, (segment, _, label) in enumerate(
            annotation.itertracks(yield_label=True)
        ):
            new_start = self.resolution * int(segment.start / self.resolution + 0.5)
            new_end = self.resolution * int(segment.end / self.resolution + 0.5)
            new_segment = Segment(start=new_start, end=new_end)
            new_annotation[new_segment, new_track] = label

        support = current_file["annotated"].extent()
        return new_annotation.support().crop(support)

