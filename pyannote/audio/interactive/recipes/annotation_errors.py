import os
from typing import Any, Dict, Iterable

import prodigy
import torch.nn.functional as F

from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from pyannote.database import util
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis

from ..utils import (
    SAMPLE_RATE,
    chunks,
    normalize,
    remove_audio_before_db,
    to_audio_spans,
    to_base64,
)


def annotation_errors_stream(
    reference: dict,
    hypothesis: dict,
    chunk: float = 30.0,
) -> Iterable[Dict]:

    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    for file in reference.keys():

        path = file
        text = file
        fileInfo = {"uri": text, "audio": path}

        duration = raw_audio.get_duration(fileInfo)
        fileInfo["duration"] = duration

        identificationErrorAnalysis = IdentificationErrorAnalysis()
        errors = identificationErrorAnalysis.difference(
            reference[file], hypothesis[file]
        )
        newLabels = {}
        for labels in errors.labels():
            a, b, c = labels
            newLabels[(a, b, c)] = a
        errors = errors.rename_labels(newLabels)
        errors = errors.subset(["correct"], invert=True)

        if duration <= chunk:
            waveform, sr = raw_audio.crop(file, Segment(0, duration))
            waveform = waveform.numpy().T
            task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
            audio_spans = to_audio_spans(errors)

            yield {
                "path": path,
                "text": text,
                "audio": task_audio,
                "audio_spans": audio_spans,
                "reference": reference[file],
                "hypothesis": hypothesis[file],
                "chunk": {"start": 0, "end": duration},
                "meta": {"file": text},
            }
        else:
            list_focus = []
            for focus in chunks(duration, chunk=chunk, shuffle=False):
                list_focus.append([errors.crop(focus, mode="intersection"), focus])

            list_focus = sorted(
                list_focus,
                key=lambda k: max(
                    (k[0].label_duration(e) for e in k[0].labels()), default=0
                ),
                reverse=True,
            )

            for seg in list_focus:
                focus = seg[1]
                task_text = f"{text} [{focus.start:.1f}, {focus.end:.1f}]"
                waveform, sr = raw_audio.crop(file, focus)
                if waveform.shape[1] != SAMPLE_RATE * chunk:
                    waveform = F.pad(
                        input=waveform,
                        pad=(0, int(SAMPLE_RATE * chunk - waveform.shape[1])),
                        mode="constant",
                        value=0,
                    )
                waveform = waveform.numpy().T
                task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
                audio_spans = to_audio_spans(seg[0], focus=focus)
                ref = reference[file].crop(focus, mode="intersection")
                ref = to_audio_spans(ref, focus=focus)
                hyp = hypothesis[file].crop(focus, mode="intersection")
                hyp = to_audio_spans(hyp, focus=focus)

                yield {
                    "path": path,
                    "text": task_text,
                    "audio": task_audio,
                    "audio_spans": audio_spans,
                    "reference": ref,
                    "hypothesis": hyp,
                    "meta": {
                        "file": text,
                        "start": f"{focus.start:.1f}",
                        "end": f"{focus.end:.1f}",
                    },
                }


@prodigy.recipe(
    "audio.errors",
    dataset=("Dataset to save annotations to", "positional", None, str),
    reference=("Path to reference file", "positional", None, str),
    hypothesis=("Path to hypothesis file ", "positional", None, str),
    chunk=(
        "Split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    precision=("Cursor speed", "option", None, int),
    beep=("Beep when the player reaches the end of a region.", "flag", None, bool),
)
def annotation_errors(
    dataset: str,
    reference: str,
    hypothesis: str,
    chunk: float = 30.0,
    precision: int = 100,
    beep: bool = False,
) -> Dict[str, Any]:

    dirname = os.path.dirname(os.path.realpath(__file__))
    pathControler = dirname + "/../errorControler.js"
    pathWave = dirname + "/../wavesurfer.js"
    pathRegion = dirname + "/../regions.js"
    pathTemplate = dirname + "/../htmltemplate.html"
    pathCss = dirname + "/../template.css"
    with open(pathControler) as txt, open(pathWave) as wave, open(
        pathRegion
    ) as region, open(pathTemplate) as html, open(pathCss) as css:
        script_text = wave.read()
        script_text += "\n" + region.read()
        script_text += "\n" + txt.read()
        templateH = html.read()
        templateC = css.read()

    prodigy.log("RECIPE: Starting recipe voice_activity_detection", locals())

    ref = util.load_rttm(reference)
    hyp = util.load_rttm(hypothesis)

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": annotation_errors_stream(ref, hyp, chunk=chunk),
        "before_db": remove_audio_before_db,
        "config": {
            "global_css": templateC,
            "javascript": script_text,
            "precision": precision,
            "beep": beep,
            "show_audio_minimap": False,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "blocks": [
                {
                    "view_id": "audio",
                },
                {"view_id": "html", "html_template": templateH},
            ],
            "show_audio_timeline": True,
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
