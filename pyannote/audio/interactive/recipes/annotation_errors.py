import os
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import prodigy
import torch.nn.functional as F
from prodigy.components.loaders import Audio as AudioLoader

from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation, Segment, Timeline
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
    source: Path,
    reference: dict,
    hypothesis: dict,
    diarization: bool = False,
    chunk: float = 30.0,
    minduration: int = 200,
) -> Iterable[Dict]:

    raw_audio = Audio(sample_rate=SAMPLE_RATE, mono=True)

    # TODO : loop on sorted chunk from all the wav files from source
    # Source as one file
    for audio_source in AudioLoader(source):

        path = audio_source["path"]
        text = audio_source["text"]
        name = audio_source["meta"]["file"]
        file = {"uri": text, "audio": path}

        duration = raw_audio.get_duration(file)
        file["duration"] = duration

        ref = reference[name]
        hyp = hypothesis[name]

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
        t = Timeline()
        for s in errors.itersegments():
            if s.duration * 1000 <= minduration:
                t.add(s)
        errors = errors.extrude(t, "strict")

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
                "reference": reference[name],
                "hypothesis": hypothesis[name],
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
                ref = reference[name].crop(focus, mode="intersection")
                ref = to_audio_spans(ref, focus=focus)
                hyp = hypothesis[name].crop(focus, mode="intersection")
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
)
def annotation_errors(
    dataset: str,
    source: Union[str, Iterable[dict]],
    reference: str,
    hypothesis: str,
    chunk: float = 30.0,
    minduration=200,
    diarization: bool = False,
) -> Dict[str, Any]:

    dirname = os.path.dirname(os.path.realpath(__file__))
    pathControler = dirname + "/../errorControler.js"
    pathWave = dirname + "/../wavesurfer.js"
    pathRegion = dirname + "/../regions.js"
    pathTemplate = dirname + "/../htmltemplate.html"
    pathLegend = dirname + "/../legend.html"
    pathCss = dirname + "/../template.css"
    with open(pathControler) as txt, open(pathWave) as wave, open(
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

    prodigy.log("RECIPE: Starting recipe voice_activity_detection", locals())

    ref = util.load_rttm(reference)
    hyp = util.load_rttm(hypothesis)

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": annotation_errors_stream(
            source,
            ref,
            hyp,
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
            "custom_theme": {"cardMinHeight": 400},
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
