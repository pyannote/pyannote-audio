#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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

from pathlib import Path
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.core import Annotation
from pyannote.database.util import load_rttm


class RemoveNonSpeech:
    """Remove (precomputed) non-speech regions from annotation

    This is useful for speaker verification databases where files contain only
    one speaker but actual speech regions are not annotated.

    Parameters
    ----------
    sad_rttm : Path
        Path to RTTM file containing speech activity detection result.

    Usage
    -----
    1. Prepare a file containing speech activity detection results in RTTM
       format (here /path/to/sad.rttm)
    2. Add a "preprocessors" section in your "config.yml" configuration file
        --------------------------------------------------------------------
        preprocessors:
            annotation:
                name: pyannote.audio.preprocessors.RemoveNonSpeech
                params:
                    sad_rttm: /path/to/sad.rttm
        --------------------------------------------------------------------
    3. Enjoy your updated "annotation" key (where non-speech regions are
       removed).
    """

    def __init__(self, sad_rttm: Path = None):
        self.sad_rttm = sad_rttm
        self.sad_ = load_rttm(self.sad_rttm)

    def __call__(self, current_file: ProtocolFile) -> Annotation:

        # get speech regions as Annotation instances
        speech_regions = self.sad_.get(current_file['uri'], Annotation())

        # remove non-speech regions from current annotation
        # aka only keep speech regions
        try:
            annotation = current_file['annotation']
            return annotation.crop(speech_regions.get_timeline())

        # this haapens when current_file has no "annotation" key
        # (e.g. for file1 and file2 in speaker verification trials)
        # in that case, we return speech regions directly
        except KeyError:
            return speech_regions
