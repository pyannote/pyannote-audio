# The MIT License (MIT)
#
# Copyright (c) 2021-2022 CNRS
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

# AUTHORS
# Jim Petiot
# Hervé Bredin

import base64
import random
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, List, Union

import prodigy
from prodigy import set_hashes
from pyannote.core import Annotation, Segment

from pyannote.audio import Audio, Pipeline

from ..common.utils import AudioForProdigy, before_db, get_audio_spans, get_chunks


def stream(
    pipeline: Pipeline,
    source: Path,
    labels: List[str],
    chunk: float = 30.0,
    randomize: bool = False,
) -> Iterable[Dict]:
    """Stream for `pyannote.audio` recipe

    Applies pretrained pipeline and sends the results for manual correction

    Parameters
    ----------
    pipeline : Pipeline
        Pretrained pipeline.
    source : Path
        Directory containing audio files to process.
    labels : list of string
        List of expected pipeline labels.
    chunk : float, optional
        Duration of chunks, in seconds. Defaults to 30s.

    Yields
    ------
    task : dict
        Prodigy task with the following keys:
        "path" : path to audio file
        "chunk" : chunk start and end times
        "audio" : base64 encoding of audio chunk
        "text" : chunk identifier "{filename} [{start} {end}]"
        "audio_spans" : list of audio spans {"start": ..., "end": ..., "label": ...}
        "audio_spans_original" : deep copy of "audio_spans"
        "meta" : metadata displayed in Prodigy UI {"file": ..., "start": ..., "end": ...}
        "config": {"labels": list of labels}
    """

    context = getattr(pipeline, "context", 2.5)

    audio_for_prodigy = AudioForProdigy()
    audio_for_pipeline = Audio(mono=True)

    chunks = get_chunks(source, chunk_duration=chunk)
    if randomize:
        chunks = list(chunks)
        random.shuffle(chunks)

    for file, excerpt in chunks:

        path = file["path"]
        filename = file["text"]
        text = f"{filename} [{excerpt.start:.1f} - {excerpt.end:.1f}]"

        # load contextualized audio excerpt
        excerpt_with_context = Segment(
            start=excerpt.start - context, end=excerpt.end + context
        )
        waveform_with_context, sample_rate = audio_for_pipeline.crop(
            path, excerpt_with_context, mode="pad"
        )

        # run pipeline on contextualized audio excerpt
        output: Annotation = pipeline(
            {"waveform": waveform_with_context, "sample_rate": sample_rate}
        )

        # crop, shift, and format output for visualization in Prodigy
        audio_spans = get_audio_spans(
            output, excerpt, excerpt_with_context=excerpt_with_context
        )

        # load audio excerpt for visualization in Prodigy
        audio = audio_for_prodigy.crop(path, excerpt)

        yield {
            "path": path,
            "text": text,
            "audio": audio,
            "audio_spans": audio_spans,
            "audio_spans_original": deepcopy(audio_spans),
            "chunk": {"start": excerpt.start, "end": excerpt.end},
            "config": {"labels": sorted(set(output.labels()) | set(labels))},
            "meta": {
                "file": filename,
                "start": f"{excerpt.start:.1f}",
                "end": f"{excerpt.end:.1f}",
            },
        }


@prodigy.recipe(
    "pyannote.audio",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=(
        "Path to directory containing audio files to annotate",
        "positional",
        None,
        Path,
    ),
    pipeline=(
        "Pretrained pipeline (name on Huggingface Hub, path to YAML file)",
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
    num_classes=(
        "Set maximum number of classes for pipelines whose number of classes is not predefined (e.g. pyannote/speaker-diarization)",
        "option",
        None,
        int,
    ),
    precision=("Keyboard temporal precision, in milliseconds.", "option", None, int),
    beep=(
        "Beep when the player reaches the end of a region.",
        "flag",
        None,
        bool,
    ),
)
def pipeline(
    dataset: str,
    source: Path,
    pipeline: Union[str, Iterable[dict]],
    chunk: float = 10.0,
    num_classes: int = 4,
    precision: int = 200,
    beep: bool = False,
) -> Dict[str, Any]:

    pipeline = Pipeline.from_pretrained(pipeline, use_auth_token=True)
    classes = pipeline.classes()

    if isinstance(classes, Iterator):
        labels = [x for _, x in zip(range(num_classes), classes)]
    else:
        labels = classes

    recipe_dir = Path(__file__).resolve().parent
    common_dir = recipe_dir.parent / "common"
    controller_js = common_dir / "controller.js"
    with open(controller_js) as txt:
        javascript = txt.read()

    # TODO: improve this part
    template = common_dir / "instructions.html"
    png = common_dir / "commands.png"
    _, instructions_html = mkstemp(text=True)
    with open(instructions_html, "w") as instructions_f, open(
        template, "r"
    ) as fp_tpl, open(png, "rb") as fp_png:
        b64 = base64.b64encode(fp_png.read()).decode("utf-8")
        instructions_f.write(fp_tpl.read().replace("{IMAGE}", b64))

    hashed_stream = (
        set_hashes(eg, input_keys=("path", "chunk"))
        for eg in stream(pipeline, source, labels, chunk=chunk, randomize=False)
    )

    return {
        "view_id": "audio_manual",
        "dataset": dataset,
        "stream": hashed_stream,
        "before_db": before_db,
        "config": {
            "javascript": javascript,
            "instructions": instructions_html,
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
            "labels": labels,
            "precision": precision,
            "beep": beep,
        },
    }
