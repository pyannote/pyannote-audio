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

from pyannote.audio.utils.signal import Binarize
from pyannote.database import ProtocolFile
from pyannote.core import Annotation

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
        self._binarize = Binarize()

    def __call__(self, current_file: ProtocolFile) -> Annotation:
        annotation = current_file["annotation"]
        support = current_file["annotated"].extent()
        return self._binarize(annotation.discretize(support, resolution=self.resolution)).crop(support)

