import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import prodigy
import torch.nn.functional as F
from prodigy.components.loaders import Audio as AudioLoader

from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation, Segment
from pyannote.database import util
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis

from ..common.utils import (
    SAMPLE_RATE,
    chunks,
    normalize,
    remove_audio_before_db,
    to_audio_spans,
    to_base64,
)


def diff_stream(
    source: Path,
    reference: dict,
    hypothesis: dict,
    listerrors: List,
    diarization: bool = False,
    chunk: float = 30.0,
    minduration: int = 200,
) -> Iterable[Dict]:

    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    # TODO : loop on sorted chunk from all the wav files from source
    if os.path.isdir(source):
        listFiles = AudioLoader(source)
    else:
        name = os.path.basename(source).rsplit(".", 1)[0]
        listFiles = [{"path": source, "text": name, "meta": {"file": source}}]

    for audio_source in listFiles:

        path = audio_source["path"]
        text = audio_source["text"]
        file = {"uri": text, "audio": path}

        duration = raw_audio.get_duration(file)
        file["duration"] = duration

        ref = reference[text]
        hyp = hypothesis[text]

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
                "reference": reference[text],
                "hypothesis": hypothesis[text],
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
                refe = ref.crop(focus, mode="intersection")
                refe = to_audio_spans(refe, focus=focus)
                hypo = hyp.crop(focus, mode="intersection")
                hypo = to_audio_spans(hypo, focus=focus)

                yield {
                    "path": path,
                    "text": task_text,
                    "audio": task_audio,
                    "audio_spans": audio_spans,
                    "reference": refe,
                    "hypothesis": hypo,
                    "meta": {
                        "file": text,
                        "start": f"{focus.start:.1f}",
                        "end": f"{focus.end:.1f}",
                    },
                }


@prodigy.recipe(
    "pyannote.review",
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

    dirname = os.path.dirname(os.path.realpath(__file__))
    pathController = dirname + "/controller.js"
    pathWave = dirname + "/../common/wavesurfer.js"
    pathRegion = dirname + "/../common/regions.js"
    pathTemplate = dirname + "/../common/template.html"
    pathLegend = dirname + "/legend.html"
    pathCss = dirname + "/../common/template.css"
    with open(pathController) as txt, open(pathWave) as wave, open(
        pathRegion
    ) as region, open(pathTemplate) as html, open(pathLegend) as legend, open(
        pathCss
    ) as css:
        script_text = wave.read()
        script_text += "\n" + region.read()
        script_text += "\n" + txt.read()
        templateH = html.read()
        legend = legend.read()
        templateC = css.read()

    ref = util.load_rttm(reference)
    hyp = util.load_rttm(hypothesis)
    listerrors = [falsealarm, confusion, misseddetection]

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": diff_stream(
            source,
            ref,
            hyp,
            listerrors,
            diarization=diarization,
            chunk=chunk,
            minduration=minduration,
        ),
        "before_db": remove_audio_before_db,
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
                {"view_id": "html", "html_template": legend},
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
