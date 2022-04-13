import base64
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, List

import prodigy
from prodigy import set_hashes
from prodigy.util import split_string
from pyannote.core import Segment
from pyannote.database import util

from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import SpeakerDiarization

from ..common.utils import (
    AudioForProdigy,
    before_db,
    get_audio_spans,
    get_chunks,
    source_to_files,
)


def review_stream(
    source: Path,
    annotations: [dict],
    labels: [dict],
    diarization: bool = False,
    chunk: float = 30.0,
) -> Iterable[Dict]:

    files = source_to_files(source)
    audio_for_prodigy = AudioForProdigy()
    audio_for_pipeline = Audio(mono=True)
    chunks = get_chunks(source, chunk_duration=chunk)
    chunks = list(chunks)

    files_annotations = {}
    for file in files:
        filename = file["text"]
        if diarization:
            list_annotations = [annotations[0][filename]] + [
                SpeakerDiarization.optimal_mapping(
                    annotations[0][filename], ann[filename]
                )
                for ann in annotations[1:]
            ]
            labels = [label for ann in list_annotations for label in ann.labels()]
            labels = list(dict.fromkeys(labels))
            files_annotations[filename] = (list_annotations, labels)
        else:
            list_annotations = [ann[filename] for ann in annotations]
            files_annotations[filename] = (list_annotations, labels)

    for file, excerpt in chunks:
        path = file["path"]
        filename = file["text"]
        text = f"{filename} [{excerpt.start:.1f} - {excerpt.end:.1f}]"
        waveform, sample_rate = audio_for_pipeline.crop(path, excerpt, mode="pad")
        audio = audio_for_prodigy.crop(path, excerpt)
        duration = audio_for_pipeline.get_duration(path)
        list_spans = []
        for annotation in files_annotations[filename][0]:
            spans = get_audio_spans(annotation, excerpt, Segment(0, duration))
            list_spans.append(spans)

        labels = files_annotations[filename][1]

        yield {
            "path": path,
            "text": text,
            "audio": audio,
            "audio_spans": [],
            "annotations": list_spans,
            "config": {"labels": labels},
            "chunk": {"start": excerpt.start, "end": excerpt.end},
            "meta": {
                "file": filename,
                "start": f"{excerpt.start:.1f}",
                "end": f"{excerpt.end:.1f}",
            },
        }


@prodigy.recipe(
    "pyannote.review",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Path to directory containing audio files whose annotation is to be checked",
        "positional",
        None,
        Path,
    ),
    annotations=(
        "Comma-separated paths to annotation files",
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
        "Optimal one-to-one mapping between the first annotation and the others",
        "flag",
        None,
        bool,
    ),
    precision=("Cursor speed", "option", None, int),
    beep=("Beep when the player reaches the end of a region.", "flag", None, bool),
)
def review(
    dataset: str,
    source: Path,
    annotations: [List[str]],
    chunk: float = 30.0,
    diarization: bool = False,
    precision: int = 200,
    beep: bool = False,
) -> Dict[str, Any]:

    recipe_dir = Path(__file__).resolve().parent
    common_dir = recipe_dir.parent / "common"

    controllerReview = recipe_dir / "controller.js"
    controller = common_dir / "controller.js"
    wavesurfer = common_dir / "wavesurfer.js"
    regions = common_dir / "regions.js"
    html = common_dir / "template.html"
    css = common_dir / "template.css"

    template = common_dir / "instructions.html"
    png = common_dir / "commands.png"
    _, instructions_html = mkstemp(text=True)
    with open(controllerReview) as sc_review, open(wavesurfer) as s_wavesurfer, open(
        regions
    ) as s_regions, open(html) as f_html, open(css) as f_css, open(
        controller
    ) as sc, open(
        png, "rb"
    ) as fp_png, open(
        instructions_html, "w"
    ) as instructions_f, open(
        template, "r"
    ) as fp_tpl:
        script_text = s_wavesurfer.read()
        script_text += "\n" + s_regions.read()
        script_text += "\n" + sc_review.read()
        script_text += "\n" + sc.read()
        templateH = f_html.read()
        templateC = f_css.read()
        b64 = base64.b64encode(fp_png.read()).decode("utf-8")
        instructions_f.write(fp_tpl.read().replace("{IMAGE}", b64))

    list_annotations = [util.load_rttm(annotation) for annotation in annotations]

    labels = [
        label
        for ann in list_annotations
        for anno in list(ann.values())
        for label in anno.labels()
    ]

    hashed_stream = (
        set_hashes(eg, input_keys=("path", "chunk"))
        for eg in review_stream(
            source, list_annotations, labels, diarization=diarization, chunk=chunk
        )
    )

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": hashed_stream,
        "before_db": before_db,
        "config": {
            "global_css": templateC,
            "javascript": script_text,
            "instructions": instructions_html,
            "precision": precision,
            "beep": beep,
            "show_audio_minimap": False,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "number_annotations": len(annotations),
            "labels": labels,
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
