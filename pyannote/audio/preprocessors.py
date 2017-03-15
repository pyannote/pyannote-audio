#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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

import xml.dom.minidom
from pyannote.database.util import FileFinder
from pyannote.core import Segment, Annotation


class GregoryGellySAD(object):

    def __init__(self, sad_yml=None):
        super(GregoryGellySAD, self).__init__()
        self.sad_yml = sad_yml
        self.file_finder_ = FileFinder(self.sad_yml)

    def __call__(self, item):

        speaker = item['speaker']
        annotation = Annotation()

        sad_xml = self.file_finder_(item)
        with open(sad_xml, 'r') as fp:
            content = xml.dom.minidom.parse(fp)
        segments = content.getElementsByTagName('SpeechSegment')

        for segment in segments:
            start = float(segment.getAttribute('stime'))
            end = float(segment.getAttribute('etime'))
            annotation[Segment(start, end)] = speaker

        return annotation
