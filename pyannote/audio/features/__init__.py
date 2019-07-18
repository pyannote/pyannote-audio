#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2018 CNRS

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

"""
# Feature extraction
"""


try:
    from .with_yaafe import YaafeCompound, YaafeZCR, YaafeMFCC
except ModuleNotFoundError as e:
    if e.args[0] == "No module named 'yaafelib'":
        msg = (
            'Feature extractors based on "yaafe" are not available '
            'because "yaafelib" could not be found.')
        print(msg)

try:
    from .with_librosa import LibrosaMFCC, LibrosaSpectrogram, LibrosaMelSpectrogram
except Exception as e:
        msg = (
            f'Feature extractors based on "librosa" are not available '
            f'because something went wrong when importing them: "{e}".')
        print(msg)

try:
    from .with_python_speech_features import PySpeechFeaturesMFCC
except Exception as e:
        msg = (
            f'Feature extractors based on "python_speech_features" are not available '
            f'because something went wrong when importing them: "{e}".')
        print(msg)

from .precomputed import Precomputed
from .precomputed import PrecomputedHTK

try:
    from .utils import RawAudio
except Exception as e:
    msg = f'Loading raw audio might fail because something went wrong: {e}.'
    print(msg)
