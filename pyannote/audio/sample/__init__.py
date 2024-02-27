# MIT License
#
# Copyright (c) 2024- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm

from pyannote.audio.core.io import Audio, AudioFile


def _sample() -> AudioFile:
    sample_wav = Path(__file__).parent / "sample.wav"
    uri = "sample"

    audio = Audio()
    waveform, sample_rate = audio(sample_wav)

    sample_rttm = Path(__file__).parent / "sample.rttm"

    annotation: Annotation = load_rttm(sample_rttm)[uri]
    duration = audio.get_duration(sample_wav)

    annotated: Timeline = Timeline([Segment(0.0, duration)], uri=uri)

    return {
        "audio": sample_wav,
        "uri": "sample",
        "waveform": waveform,
        "sample_rate": sample_rate,
        "annotation": annotation,
        "annotated": annotated,
    }


SAMPLE_FILE = _sample()
