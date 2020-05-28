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

# types
from pathlib import Path
from typing import Text, Dict, List, Iterable
from pyannote.core import Segment, Timeline, Annotation

# prodigy
import prodigy
from prodigy.components.loaders import Audio
from prodigy.components.db import connect
from .utils import SAMPLE_RATE
from .utils import normalize
from .utils import to_base64
from .utils import to_audio_spans
from .utils import chunks
from .utils import remove_audio_before_db
from copy import deepcopy

# pyannote.audio
from ..pipeline import InteractiveDiarization
from ..pipeline import PRETRAINED_PARAMS
from pyannote.audio.features import RawAudio
from pyannote.audio.features.utils import get_audio_duration


def sad_manual_stream(
    pipeline: InteractiveDiarization, source: Path, chunk: float = 10.0
) -> Iterable[Dict]:
    """Stream for pyannote.sad.manual recipe

    Applies (pretrained) speech activity detection and sends the results for
    manual correction chunk by chunk.

    Parameters
    ----------
    pipeline : InteractiveDiarization
        Pretrained speaker diarization interactive pipeline.
        Note that only the speech activity detection part is used.
    source : Path
        Directory containing audio files to process.
    chunk : float, optional
        Duration of chunks, in seconds. Defaults to 10s.

    Yields
    ------
    task : dict
        Prodigy task with the following keys:
        "path" : path to audio file
        "text" : name of audio file
        "chunk" : chunk start and end times
        "audio" : base64 encoding of audio chunk
        "audio_spans" : speech spans detected by pretrained SAD model
        "audio_spans_original" : copy of "audio_spans"
        "meta" : additional meta-data displayed in Prodigy UI
        "recipe" : "pyannote.sad.manual"

    """

    raw_audio = RawAudio(sample_rate=SAMPLE_RATE, mono=True)

    for audio_source in Audio(source):

        path = audio_source["path"]
        text = audio_source["text"]
        file = {"uri": text, "database": source, "audio": path}

        duration = get_audio_duration(file)
        file["duration"] = duration

        prodigy.log(f"RECIPE: detecting speech regions in '{path}'")

        speech: Annotation = pipeline.compute_speech(file).to_annotation(
            generator=iter(lambda: "SPEECH", None)
        )

        if duration <= chunk:
            waveform = raw_audio.crop(file, Segment(0, duration))
            task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
            task_audio_spans = to_audio_spans(speech)

            yield {
                "path": path,
                "text": text,
                "audio": task_audio,
                "audio_spans": task_audio_spans,
                "audio_spans_original": deepcopy(task_audio_spans),
                "chunk": {"start": 0, "end": duration},
                "meta": {"file": text},
                # this is needed by other recipes
                "recipe": "pyannote.sad.manual",
            }

        else:
            for focus in chunks(duration, chunk=chunk, shuffle=True):
                task_text = f"{text} [{focus.start:.1f}, {focus.end:.1f}]"
                waveform = raw_audio.crop(file, focus)
                task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
                task_audio_spans = to_audio_spans(
                    speech.crop(focus, mode="intersection"), focus=focus
                )

                yield {
                    "path": path,
                    "text": task_text,
                    "audio": task_audio,
                    "audio_spans": task_audio_spans,
                    "audio_spans_original": deepcopy(task_audio_spans),
                    "chunk": {"start": focus.start, "end": focus.end},
                    "meta": {
                        "file": text,
                        "start": f"{focus.start:.1f}",
                        "end": f"{focus.end:.1f}",
                    },
                    # this is needed by other recipes
                    "recipe": "pyannote.sad.manual",
                }


def sad_manual_before_db(examples: List[Dict]) -> List[Dict]:
    """Remove 'audio' key and shift spans back to t0=0 base

    Parameters
    ----------
    examples : list of dict
        Examples.

    Returns
    -------
    examples : list of dict
        Examples with "audio" key removed and shifted.
    """

    examples = remove_audio_before_db(examples)

    for eg in examples:

        # shift audio spans back to the t0=0 base
        chunk = eg.get("chunk", None)
        if chunk is not None:
            start = chunk["start"]
            for span in eg["audio_spans"]:
                span["start"] += start
                span["end"] += start
            for span in eg["audio_spans_original"]:
                span["start"] += start
                span["end"] += start

    return examples


@prodigy.recipe(
    "pyannote.sad.manual",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Directory containing audio files to annotate", "positional", None, Path),
    chunk=(
        "split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    speed=(
        "set the playback rate (0.5 means half the normal speed, 2 means double speed and so on)",
        "option",
        None,
        float,
    ),
)
def sad_manual(
    dataset: Text, source: Path, chunk: float = 10.0, speed: float = 1.0
) -> Dict:

    pipeline = InteractiveDiarization().instantiate(PRETRAINED_PARAMS)

    return {
        "dataset": dataset,
        "view_id": "audio_manual",
        "stream": sad_manual_stream(pipeline, source, chunk=chunk),
        "before_db": sad_manual_before_db,
        "config": {
            "audio_autoplay": True,
            "audio_loop": True,
            "show_audio_minimap": False,
            "audio_bar_width": 3,
            "audio_bar_height": 1,
            "audio_rate": speed,
            "labels": ["SPEECH",],
        },
    }


def load_sad_manual(dataset: Text, path: Text) -> Dict:
    """Load accepted pyannote.sad.manual examples

    Parameters
    ----------
    dataset : str
        Dataset containing annotations.
    path : str
        Path to annotated file

    Returns
    -------
    file : dict
        Dictionary containing the following keys:
        "audio" (Path) : path to audio file
        "annotated" (Timeline) : part of the audio annotated and accepted
        "speech" (Timeline) : part of the audio accepted as speech
    """

    db = connect()

    examples = [
        eg
        for eg in db.get_dataset(dataset)
        if eg["recipe"] == "pyannote.sad.manual"
        and eg["path"] == path
        and eg["answer"] == "accept"
    ]

    speech = Timeline(
        segments=[
            Segment(span["start"], span["end"])
            for eg in examples
            for span in eg["audio_spans"]
        ],
    ).support()

    annotated = Timeline(segments=[Segment(**eg["chunk"]) for eg in examples]).support()

    prodigy.log(f"RECIPE: {path}: loaded speech regions")

    return {
        "audio": Path(path),
        "speech": speech,
        "annotated": annotated,
    }
