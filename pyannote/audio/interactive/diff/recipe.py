import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import prodigy
from prodigy import set_hashes
from prodigy.components.loaders import Audio as AudioLoader
from pyannote.core import Annotation, Segment
from pyannote.database import util
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis

from pyannote.audio import Audio
from pyannote.audio.pipelines import SpeakerDiarization

from ..common.utils import AudioForProdigy, before_db, get_audio_spans, get_chunks


def diff_stream(
    source: Path,
    reference: dict,
    hypothesis: dict,
    listerrors: List,
    diarization: bool = False,
    chunk: float = 30.0,
    minduration: int = 200,
) -> Iterable[Dict]:

    if os.path.isdir(source):
        files = AudioLoader(source)
    else:
        name = os.path.basename(source).rsplit(".", 1)[0]
        files = [{"path": source, "text": name, "meta": {"file": source}}]

    files_errors = {}
    for file in files:
        filename = file["text"]
        ref = reference[filename]
        hyp = hypothesis[filename]
        if diarization:
            hyp: Annotation = SpeakerDiarization.optimal_mapping(ref, hyp)
        identificationErrorAnalysis = IdentificationErrorAnalysis()
        errors = identificationErrorAnalysis.difference(ref, hyp)
        newLabels = {}
        for labels in errors.labels():
            a, b, c = labels
            newLabels[(a, b, c)] = a
        errors = errors.rename_labels(newLabels)
        errors = errors.subset(["correct"], invert=True)
        if listerrors[0] or listerrors[1] or listerrors[2]:
            if not listerrors[0]:
                errors = errors.subset(["false alarm"], invert=True)
            if not listerrors[1]:
                errors = errors.subset(["confusion"], invert=True)
            if not listerrors[2]:
                errors = errors.subset(["missed detection"], invert=True)
        clean_errors = Annotation()
        for segment, track, label in errors.itertracks(yield_label=True):
            if segment.duration * 1000 > minduration:
                clean_errors[segment, track] = label
        errors = clean_errors
        files_errors[filename] = errors

    audio_for_prodigy = AudioForProdigy()
    audio_for_pipeline = Audio(mono=True)
    chunks = get_chunks(source, chunk_duration=chunk)
    chunks = list(chunks)
    chunks = sorted(
        chunks,
        key=lambda k: max(
            (
                files_errors[k[0]["text"]]
                .crop(k[1], mode="intersection")
                .label_duration(e)
                for e in files_errors[k[0]["text"]]
                .crop(k[1], mode="intersection")
                .labels()
            ),
            default=0,
        ),
        reverse=True,
    )

    for file, excerpt in chunks:
        path = file["path"]
        filename = file["text"]
        text = f"{filename} [{excerpt.start:.1f} - {excerpt.end:.1f}]"
        duration = audio_for_pipeline.get_duration(path)
        # load audio excerpt
        waveform, sample_rate = audio_for_pipeline.crop(path, excerpt, mode="pad")
        # load audio excerpt for visualization in Prodigy
        audio = audio_for_prodigy.crop(path, excerpt)

        audio_spans = get_audio_spans(
            files_errors[filename], excerpt, Segment(0, duration)
        )
        ref = get_audio_spans(reference[filename], excerpt, Segment(0, duration))
        hyp = get_audio_spans(hypothesis[filename], excerpt, Segment(0, duration))

        yield {
            "path": path,
            "text": text,
            "audio": audio,
            "audio_spans": audio_spans,
            "reference": ref,
            "hypothesis": hyp,
            "chunk": {"start": excerpt.start, "end": excerpt.end},
            "meta": {
                "file": filename,
                "start": f"{excerpt.start:.1f}",
                "end": f"{excerpt.end:.1f}",
            },
        }


@prodigy.recipe(
    "pyannote.diff",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Path to directory containing audio files whose annotation is to be checked",
        "positional",
        None,
        str,
    ),
    reference=("Path to reference file", "positional", None, str),
    hypothesis=("Path to hypothesis file ", "positional", None, str),
    chunk=(
        "Split long audio files into shorter chunks of that many seconds each",
        "option",
        None,
        float,
    ),
    minduration=("Minimum duration of errors in ms", "option", None, int),
    diarization=(
        "Optimal one-to-one mapping between reference and hypothesis",
        "flag",
        None,
        bool,
    ),
    falsealarm=("Display false alarm errors", "flag", None, bool),
    confusion=("Display confusion errors", "flag", None, bool),
    misseddetection=("Display missed detection errors", "flag", None, bool),
)
def diff(
    dataset: str,
    source: Union[str, Iterable[dict]],
    reference: str,
    hypothesis: str,
    chunk: float = 30.0,
    minduration=200,
    diarization: bool = False,
    falsealarm: bool = False,
    confusion: bool = False,
    misseddetection: bool = False,
) -> Dict[str, Any]:

    recipe_dir = Path(__file__).resolve().parent
    common_dir = recipe_dir.parent / "common"

    controllerDiff = recipe_dir / "controller.js"
    legend = recipe_dir / "legend.html"
    wavesurfer = common_dir / "wavesurfer.js"
    regions = common_dir / "regions.js"
    html = common_dir / "template.html"
    css = common_dir / "template.css"

    with open(controllerDiff) as sc_diff, open(wavesurfer) as s_wavesurfer, open(
        regions
    ) as s_regions, open(html) as f_html, open(legend) as f_legend, open(css) as f_css:
        script_text = s_wavesurfer.read()
        script_text += "\n" + s_regions.read()
        script_text += "\n" + sc_diff.read()
        templateH = f_html.read()
        templateL = f_legend.read()
        templateC = f_css.read()

    ref = util.load_rttm(reference)
    hyp = util.load_rttm(hypothesis)
    listerrors = [falsealarm, confusion, misseddetection]

    hstream = (
        set_hashes(eg, input_keys=("path", "chunk"))
        for eg in diff_stream(
            source,
            ref,
            hyp,
            listerrors,
            diarization=diarization,
            chunk=chunk,
            minduration=minduration,
        )
    )

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": hstream,
        "before_db": before_db,
        "config": {
            "global_css": templateC,
            "javascript": script_text,
            "show_audio_minimap": False,
            "audio_bar_width": 0,
            "audio_bar_height": 1,
            "custom_theme": {
                "cardMinHeight": 400,
                "labels": {
                    "false alarm": "#9932cc",
                    "confusion": "#ff6347",
                    "missed detection": "#00ffff",
                },
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
                },
            },
            "blocks": [
                {"view_id": "html", "html_template": templateL},
                {"view_id": "audio"},
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
