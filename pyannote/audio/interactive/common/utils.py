# The MIT License (MIT)
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Text

import numpy as np
import scipy.io.wavfile
from prodigy.components.loaders import Audio as ProdigyAudioLoader
from pyannote.core import Annotation, Segment, SlidingWindow

from pyannote.audio import Audio


class AudioForProdigy(Audio):
    def __init__(self):
        super().__init__(sample_rate=16000, mono=True)

    def crop(self, path: Path, excerpt: Segment) -> Text:
        waveform, _ = super().crop(path, excerpt)
        waveform = waveform.numpy().T
        waveform /= np.max(np.abs(waveform)) + 1e-8
        with io.BytesIO() as content:
            scipy.io.wavfile.write(content, self.sample_rate, waveform)
            content.seek(0)
            b64 = base64.b64encode(content.read()).decode()
            b64 = f"data:audio/x-wav;base64,{b64}"
        return b64


def to_audio_spans(annotation: Annotation, focus: Segment = None) -> Dict:
    """Convert pyannote.core.Annotation to Prodigy's audio_spans

    Parameters
    ----------
    annotation : Annotation
        Annotation with t=0s time origin.
    focus : Segment, optional
        When provided, use its start time as audio_spans time origin.

    Returns
    -------
    audio_spans : list of dict
    """
    shift = 0.0 if focus is None else focus.start
    return [
        {"start": segment.start - shift, "end": segment.end - shift, "label": label}
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def get_audio_spans(
    annotation: Annotation, excerpt: Segment, excerpt_with_context: Segment = None
):

    excerpt_with_context = excerpt_with_context or excerpt
    shift = excerpt.start - excerpt_with_context.start

    shifted_excerpt = Segment(start=shift, end=shift + excerpt.duration)
    excerpt_annotation = annotation.crop(shifted_excerpt)

    return [
        {"start": segment.start - shift, "end": segment.end - shift, "label": label}
        for segment, _, label in excerpt_annotation.itertracks(yield_label=True)
    ]


def remove_audio_before_db(examples: List[Dict]) -> List[Dict]:
    """Remove (potentially heavy) 'audio' key from examples

    Parameters
    ----------
    examples : list of dict
        Examples.

    Returns
    -------
    examples : list of dict
        Examples with 'audio' key removed.
    """
    for eg in examples:
        if "audio" in eg:
            del eg["audio"]

    return examples


def source_to_files(source: Path) -> List[Dict]:
    """
    Convert a directory or a file path to a list of files object for prodigy
    """
    if source.is_dir():
        files = ProdigyAudioLoader(source)
    else:
        name = source.stem
        files = [{"path": source, "text": name, "meta": {"file": source}}]

    return files


def get_chunks(source: Path, chunk_duration: Optional[float] = None):

    files = source_to_files(source)
    audio = Audio()

    for file in files:

        duration = audio.get_duration(file["path"])

        if (chunk_duration is None) or (duration < chunk_duration):
            chunks = [Segment(0, duration)]
        else:
            windows = SlidingWindow(
                start=0.0, duration=chunk_duration, step=chunk_duration
            )
            chunks = windows(Segment(0, duration), align_last=True)

        for chunk in chunks:
            yield file, chunk


def before_db(examples):
    """Post-process examples before sending them to the database

    1. Remove "audio" key as it is very heavy and can easily be retrieved from other keys
    2. Shift Prodigy/wavesurfer chunk-based audio spans so that their timing are file-based.
    """

    for eg in examples:

        # 1. remove "audio" key as it can be retrieved from "chunk" and "path" keys
        if "audio" in eg:
            del eg["audio"]

        # 2. shift audio spans
        chunk_start = eg["chunk"]["start"]
        audio_spans_keys = [key for key in eg if "audio_spans" in key]
        for key in audio_spans_keys:
            eg[key] = [
                {
                    "start": span["start"] + chunk_start,
                    "end": span["end"] + chunk_start,
                    "label": span["label"],
                }
                for span in eg[key]
            ]

    return examples
