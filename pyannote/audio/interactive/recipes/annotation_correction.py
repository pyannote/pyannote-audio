import base64
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import prodigy
import torch.nn.functional as F
from prodigy.components.loaders import Audio as AudioLoader
from prodigy.util import split_string

from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from pyannote.database import util

from ..utils import (
    SAMPLE_RATE,
    chunks,
    normalize,
    remove_audio_before_db,
    to_audio_spans,
    to_base64,
)


def annotation_correction_stream(
    source: Path,
    annotations: [dict],
    diarization: bool = False,
    globallabels: bool = False,
    chunk: float = 10.0,
) -> Iterable[Dict]:

    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    if os.path.isdir(source):
        listFiles = AudioLoader(source)
    else:
        name = os.path.basename(source).rsplit(".", 1)[0]
        listFiles = [{"path": source, "text": name, "meta": {"file": source}}]

    for audio_source in listFiles:

        path = audio_source["path"]
        text = audio_source["text"]
        file = {"uri": text, "audio": path, "database": source}

        duration = raw_audio.get_duration(file)
        file["duration"] = duration

        if diarization:
            list_annotations = [annotations[0][text]] + [
                SpeakerDiarization.optimal_mapping(annotations[0][text], ann[text])
                for ann in annotations[1:]
            ]
        else:
            list_annotations = [ann[text] for ann in annotations]

        if duration <= chunk:
            waveform, sr = raw_audio.crop(file, Segment(0, duration))
            waveform = waveform.numpy().T
            task_audio = to_base64(normalize(waveform), sample_rate=SAMPLE_RATE)
            labels = [label for ann in list_annotations for label in ann.labels()]
            labels = list(dict.fromkeys(labels))

            yield {
                "path": path,
                "text": text,
                "audio": task_audio,
                "audio_spans": [],
                "annotations": list_annotations,
                "chunk": {"start": 0, "end": duration},
                "config": {"labels": labels},
                "meta": {"file": text},
            }
        else:
            if globallabels:
                labels = [label for ann in list_annotations for label in ann.labels()]
                labels = list(dict.fromkeys(labels))

            for focus in chunks(duration, chunk=chunk, shuffle=False):
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
                list_spans = []
                label = []
                for ann in list_annotations:
                    sa = ann.crop(focus, mode="intersection")
                    spans = to_audio_spans(sa, focus=focus)
                    list_spans.append(spans)
                    label += sa.labels()

                if not globallabels:
                    labels = list(dict.fromkeys(label))

                yield {
                    "path": path,
                    "text": task_text,
                    "audio": task_audio,
                    "audio_spans": [],
                    "annotations": list_spans,
                    "config": {"labels": labels},
                    "meta": {
                        "file": text,
                        "start": f"{focus.start:.1f}",
                        "end": f"{focus.end:.1f}",
                    },
                }


@prodigy.recipe(
    "audio.correction",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Path to directory containing audio files whose annotation is to be checked",
        "positional",
        None,
        str,
    ),
    annotations=(
        "Comma-separated paths to annotation files ",
        "positional",
        None,
        split_string,
    ),
    chunk=(
        "Split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    diarization=(
        "Optimal one-to-one mapping between reference and hypothesis",
        "flag",
        None,
        bool,
    ),
    globallabels=(
        "Shows the labels of the whole file (not chunk only)",
        "flag",
        None,
        bool,
    ),
    precision=("Cursor speed", "option", None, int),
    beep=("Beep when the player reaches the end of a region.", "flag", None, bool),
)
def annotation_correction(
    dataset: str,
    source: Union[str, Iterable[dict]],
    annotations: [List[str]],
    chunk: float = 10.0,
    diarization: bool = False,
    globallabels: bool = False,
    precision: int = 100,
    beep: bool = False,
) -> Dict[str, Any]:

    dirname = os.path.dirname(os.path.realpath(__file__))
    pathControler = dirname + "/../correctionControler.js"
    pathShortcuts = dirname + "/../wavesurferControler.js"
    pathWave = dirname + "/../wavesurfer.js"
    pathRegion = dirname + "/../regions.js"
    pathHtml = dirname + "/../instructions.html"
    png = dirname + "/../commands.png"
    pathTemplate = dirname + "/../htmltemplate.html"
    pathCss = dirname + "/../template.css"
    help = dirname + "/../help.html"
    with open(pathControler) as txt, open(pathWave) as wave, open(
        pathRegion
    ) as region, open(pathTemplate) as html, open(pathCss) as css, open(
        pathShortcuts
    ) as sc, open(
        png, "rb"
    ) as fp_png, open(
        help, "w"
    ) as fp_help, open(
        pathHtml
    ) as fp_html:
        script_text = wave.read()
        script_text += "\n" + region.read()
        script_text += "\n" + txt.read()
        script_text += "\n" + sc.read()
        templateH = html.read()
        templateC = css.read()
        b64 = base64.b64encode(fp_png.read()).decode("utf-8")
        fp_help.write(fp_html.read().replace("{IMAGE}", b64))

    prodigy.log("RECIPE: Starting recipe voice_activity_detection", locals())

    list_annotations = [util.load_rttm(annotation) for annotation in annotations]

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": annotation_correction_stream(
            source,
            list_annotations,
            diarization=diarization,
            globallabels=globallabels,
            chunk=chunk,
        ),
        "before_db": remove_audio_before_db,
        "config": {
            "global_css": templateC,
            "javascript": script_text,
            "instructions": help,
            "precision": precision,
            "beep": beep,
            "show_audio_minimap": False,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "number_annotations": len(annotations),
            "custom_theme": {
                "palettes": {
                    "audio": [
                        "#ffd700",
                        "#00ffff",
                        "#ff00ff",
                        "#00ff00",
                        "#9932cc",
                        "#00bfff",
                        "#ff7f50",
                        "#66cdaa",
                    ],
                }
            },
            "blocks": [
                {
                    "view_id": "audio_manual",
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
