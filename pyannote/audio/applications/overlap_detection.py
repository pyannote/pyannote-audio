#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

from pyannote.audio.pipeline.overlap_detection \
    import OverlapDetection as OverlapDetectionPipeline
from .speech_detection import SpeechActivityDetection
from pyannote.core import Timeline


class OverlapDetection(SpeechActivityDetection):

    Pipeline = OverlapDetectionPipeline

    def validate_init(self, protocol, subset='development'):

        validation_data = super().validate_init(protocol, subset=subset)
        for current_file in validation_data:

            uri = current_file['uri']

            # build overlap reference
            overlap = Timeline(uri=uri)
            turns = current_file['annotation']
            for track1, track2 in turns.co_iter(turns):
                if track1 == track2:
                    continue
                overlap.add(track1[0] & track2[0])
            current_file['overlap'] = overlap.support().to_annotation()
            # TODO. make 'annotated' focus on speech regions only

        return validation_data
