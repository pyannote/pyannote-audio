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
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import prodigy
from prodigy.components.loaders import Audio as AudioLoader

from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment

from ..utils import (
    SAMPLE_RATE,
    chunks,
    normalize,
    remove_audio_before_db,
    to_audio_spans,
    to_base64,
)


def voice_activity_detection_stream(
    pipeline: VoiceActivityDetection,
    source: Path,
    chunk: float = 10.0,
) -> Iterable[Dict]:
    """
    Stream for `audio.vad` recipe
`
    Applies (pretrained) speech activity detection and sends the results for
    manual correction chunk by chunk.

    Parameters
    ----------
    pipeline : VoiceActivityDetection
        Pretrained speech activity detection pipeline.
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
    """
    extend = 0.5 * pipeline.segmentation_inference_.duration
    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    for audio_source in AudioLoader(source):

        path = audio_source["path"]
        text = audio_source["text"]
        file = {"uri": text, "database": source, "audio": path}

        duration = raw_audio.get_duration(file)
        file["duration"] = duration

        if duration <= chunk:
            if pipeline is not None:
                speech: Annotation = pipeline(file)
                task_audio_spans = to_audio_spans(speech)
            else:
                task_audio_spans = []
            waveform, sr = raw_audio.crop(file, Segment(0, duration))
            waveform = waveform.numpy().T
            task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)

            yield {
                "path": path,
                "text": text,
                "audio": task_audio,
                "audio_spans": task_audio_spans,
                "audio_spans_original": deepcopy(task_audio_spans),
                "chunk": {"start": 0, "end": duration},
                "meta": {"file": text},
            }

        else:
            for focus in chunks(duration, chunk=chunk, shuffle=False):
                task_text = f"{text} [{focus.start:.1f}, {focus.end:.1f}]"
                waveform, sr = raw_audio.crop(file, focus)
                if waveform.shape[1] != SAMPLE_RATE * chunk:
                    waveform = waveform.pad(
                        input=waveform,
                        pad=(0, SAMPLE_RATE * chunk - waveform.shape[1]),
                        mode="constant",
                        value=0,
                    )
                if pipeline is not None:
                    longFocus = Segment(
                        max(0, focus.start - extend), min(focus.end + extend, duration)
                    )
                    longWaveform, sr = raw_audio.crop(file, longFocus)
                    speech: Annotation = pipeline(
                        {"waveform": longWaveform, "sample_rate": sr}
                    )
                    diffStart = focus.start - longFocus.start
                    trueFocus = Segment(diffStart, diffStart + focus.duration)
                    speech = speech.crop(trueFocus, mode="intersection")
                    task_audio_spans = to_audio_spans(speech, focus=trueFocus)
                else:
                    task_audio_spans = []
                waveform = waveform.numpy().T
                task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)

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
                }


@prodigy.recipe(
    "audio.vad",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Path to directory containing audio files to annotate",
        "positional",
        None,
        str,
    ),
    chunk=(
        "Split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    pipeline=(
        "Pretrained pipeline (name on Huggingface Hub, path to YAML file, or NONE)",
        "option",
        None,
        str,
    ),
    precision=("Keyboard temporal precision, in milliseconds.", "option", None, int),
    beep=(
        "Beep when the player reaches the end of a region.",
        "flag",
        None,
        bool,
    ),
    spectrogram=("Display spectrogram", "flag", None, bool),
)
def voice_activity_detection(
    dataset: str,
    source: Union[str, Iterable[dict]],
    chunk: float = 10.0,
    pipeline: Optional[str] = None,
    precision: int = 200,
    beep: bool = False,
    spectrogram: bool = False,
) -> Dict[str, Any]:



    if pipeline is None:
        vad = VoiceActivityDetection(segmentation="pyannote/segmentation")
        vad.instantiate({
            "onset": 0.767,
            "offset": 0.377,
            "min_duration_on": 0.136,
            "min_duration_off": 0.067,
        }

    elif pipeline.lower() == "NONE"
        vad = None
    else:
        vad = Pipeline.from_pretrained(pipeline)

    # FIXME: changing every .prodigy-content might be too much
    # FIXME: would it possible to only change the spectrogram container?
    global_css = ".prodigy-content{ text-align : left}" if spectrogram else ""

    dirname = os.path.dirname(os.path.realpath(__file__))

    controller_js = dirname + "/wavesurferControler.js"
    with open(controller_js) as txt:
        javascript = txt.read()

    template = dirname + "/instructions.html"
    png = dirname + "/commands.png"
    html = dirname + "/help.html"

    with open(html, "w") as fp, open(template, "r") as fp_tpl, open(
                png, "rb") as fp_png:
        b64 = base64.b64encode(fp_png.read()).decode("utf-8")
        fp.write(fp_tpl.read().replace("{IMAGE}", b64))

    return {
        "view_id": "audio_manual",
        "dataset": dataset,
        "stream": voice_activity_detection_stream(vad, source, chunk=chunk),
        "before_db": remove_audio_before_db,
        "config": {
            "javascript": javascript,
            "instructions": html,
            "global_css": global_css,
            "buttons": ["accept", "ignore", "undo"],
            "keymap": {
                "accept": ["enter"],
                "ignore": ["escape"],
                "undo": ["u"],
                "playpause": ["space"],
            },
            "show_audio_minimap": False,
            "audio_autoplay": True,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "show_flag": True,
            "labels": ["SPEECH"],
            "precision": precision,
            "beep": beep,
            "spectrogram": spectrogram,
        },
    }
