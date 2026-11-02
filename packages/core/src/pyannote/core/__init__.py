# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

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


import importlib.metadata

PYANNOTE_URI = "uri"
PYANNOTE_MODALITY = "modality"
PYANNOTE_SEGMENT = "segment"
PYANNOTE_TRACK = "track"
PYANNOTE_LABEL = "label"
PYANNOTE_SCORE = "score"
PYANNOTE_IDENTITY = "identity"

from .annotation import Annotation
from .feature import SlidingWindowFeature
from .segment import Segment, SlidingWindow
from .timeline import Timeline

__all__ = [
    "PYANNOTE_IDENTITY",
    "PYANNOTE_LABEL",
    "PYANNOTE_MODALITY",
    "PYANNOTE_SCORE",
    "PYANNOTE_SEGMENT",
    "PYANNOTE_TRACK",
    "PYANNOTE_URI",
    "Annotation",
    "Segment",
    "SlidingWindow",
    "SlidingWindowFeature",
    "Timeline",
]

try:
    from .notebook import notebook
except ImportError:
    pass
else:
    __all__ += ["notebook"]

Segment.set_precision()

__version__ = importlib.metadata.version("pyannote-core")
