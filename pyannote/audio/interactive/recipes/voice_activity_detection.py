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
    Stream for pyannote.voice_activity_detection recipe
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
        "split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    pipeline=(
        "Path to YAML file of pretrained pipeline",
        "option",
        None,
        str,
    ),
    precision=("Temporal precision (in milliseconds)", "option", None, int),
)
def voice_activity_detection(
    dataset: str,
    source: Union[str, Iterable[dict]],
    chunk: float = 10.0,
    pipeline: Optional[str] = None,
    precision: int = 100,
) -> Dict[str, Any]:

    if pipeline is not None:
        if pipeline.lower() in ["no", "none"]:
            vad = None
        else:
            vad = Pipeline.from_pretrained(pipeline)
    else:
        vad = VoiceActivityDetection(segmentation="pyannote/segmentation", step=0.5)
        HYPER_PARAMETERS = {
            "onset": 0.767,
            "offset": 0.377,
            "min_duration_on": 0.136,
            "min_duration_off": 0.067,
        }
        vad.instantiate(HYPER_PARAMETERS)

    pathControler = (
        os.path.dirname(os.path.realpath(__file__)) + "/wavesurferControler.js"
    )
    pathInstructions = (
        os.path.dirname(os.path.realpath(__file__)) + "/instructions.html"
    )
    pathHtml = os.path.dirname(os.path.realpath(__file__)) + "/help.html"
    with open(pathControler) as txt:
        script_text = txt.read()

    with open(pathHtml, "w") as html:
        with open(pathInstructions, "r") as i:
            with open(
                os.path.dirname(os.path.realpath(__file__)) + "/commands.png", "rb"
            ) as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
                instruction = i.read()
                instruction = instruction.replace("{IMAGE}", b64)
                html.write(instruction)

    prodigy.log("RECIPE: Starting recipe voice_activity_detection", locals())

    return {
        "view_id": "audio_manual",
        "dataset": dataset,
        "stream": voice_activity_detection_stream(vad, source, chunk=chunk),
        "before_db": remove_audio_before_db,
        "config": {
            "javascript": script_text,
            "instructions": pathHtml,
            "labels": ["Speech"],
            "audio_autoplay": True,
            "show_audio_minimap": False,
            "precision": precision,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "buttons": ["accept", "ignore", "undo"],
            "keymap": {
                "accept": ["enter"],
                "ignore": ["escape"],
                "undo": ["u"],
                "playpause": ["space"],
            },
            "show_flag": True,
        },
    }
