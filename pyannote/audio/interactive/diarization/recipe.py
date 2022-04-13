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
import functools
import random
from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import prodigy
import torch
from frozendict import frozendict
from prodigy import set_hashes
from pyannote.core import Annotation, Segment
from scipy.spatial.distance import cdist

from pyannote.audio import Audio, Inference, Pipeline

from ..common.utils import AudioForProdigy, before_db, get_audio_spans, get_chunks

# TODO: improve this part ?
inference = Inference("pyannote/embedding", window="whole")
dim = inference.model.introspection.dimension
nSpeakerVoices = np.array(
    [], dtype=[("name", "U100"), ("embedding", "f4", dim), ("nb", "i4")]
)
buffer = {}


def freezeargs(func):
    """
    Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(
            [frozendict(arg) if isinstance(arg, dict) else arg for arg in args]
        )
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def on_exit(controller):
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y")
    name = "embeddings_" + date_time
    np.save(name, nSpeakerVoices)


def update(answers):
    global nSpeakerVoices
    global buffer

    eg = answers[0]
    for span in eg["audio_spans"]:
        if span["label"] in eg:
            speaker = eg[span["label"]]
            span["label"] = eg[span["label"]]
        else:
            if span["label"].startswith("SPEAKER_"):
                speaker = "G" + span["label"]
            else:
                speaker = span["label"]

        speaker = speaker.strip()
        segment = Segment(
            span["start"] + eg["chunk"]["start"], span["end"] + eg["chunk"]["start"]
        )
        audio_for_pipeline = Audio(mono=True)
        wav, sample_rate = audio_for_pipeline.crop(eg["path"], segment, mode="pad")

        if speaker not in buffer:
            empty_waveform = torch.Tensor([])
            buffer[speaker] = [empty_waveform, 0]

        if buffer[speaker][1] + segment.duration >= 5:

            combine_waveform = torch.cat((wav, buffer[speaker][0]), dim=1)

            embedding = getEmb(combine_waveform, sample_rate)

            empty_waveform = torch.Tensor([])
            buffer[speaker] = [empty_waveform, 0]

            if not np.isnan(embedding).any():
                if speaker in nSpeakerVoices["name"]:
                    i = np.where(nSpeakerVoices["name"] == speaker)
                    i = i[0][0]
                    nSpeakerVoices[i]["embedding"] = (
                        (nSpeakerVoices[i]["nb"] * nSpeakerVoices[i]["embedding"])
                        + embedding
                    ) / (nSpeakerVoices[i]["nb"] + 1)
                    nSpeakerVoices[i]["nb"] += 1
                else:
                    size = nSpeakerVoices.size + 1
                    nSpeakerVoices.resize(size, refcheck=False)
                    nSpeakerVoices[nSpeakerVoices.size - 1] = (speaker, embedding, 1)
        else:
            combine_waveform = torch.cat((wav, buffer[speaker][0]), dim=1)
            duration = buffer[speaker][1] + segment.duration
            buffer[speaker] = [combine_waveform, duration]


def validate_answer(eg):
    for field in eg["config"]["blocks"]:
        if "field_id" in field:
            speaker = field["field_id"]
            if speaker in eg and eg[speaker] != "":
                flag = False
                for span in eg["audio_spans"]:
                    if span["label"] == speaker:
                        flag = True
                        break
                assert flag, "You have named a speaker not present in the audio"


@freezeargs
@lru_cache(maxsize=20)
def getEmb(wav, sample_rate):
    try:
        embedding = inference({"waveform": wav, "sample_rate": sample_rate})
    except (RuntimeError, ValueError):
        embedding = [float("nan")]
    return embedding


def diarization_stream(
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

        labels = sorted(set(labels) | set(output.labels()))
        new_labels = []

        # group by label
        audio_spans = sorted(audio_spans, key=lambda x: x["label"])
        for label, segments in groupby(audio_spans, key=lambda x: x["label"]):

            combine_waveform = torch.Tensor([])
            for segment in list(segments):
                waveform, sample_rate = audio_for_pipeline.crop(
                    path,
                    Segment(
                        segment["start"] + excerpt.start, segment["end"] + excerpt.start
                    ),
                )
                combine_waveform = torch.cat((combine_waveform, waveform), dim=1)

            embedding = getEmb(combine_waveform, sample_rate)

            if not np.isnan(embedding).any():
                try:
                    distances = cdist(
                        nSpeakerVoices["embedding"],
                        embedding.reshape(1, -1),
                        metric="cosine",
                    )
                    index_min = np.argmin(distances)

                except ValueError:
                    distances = [1000]
                    index_min = 0

                # TODO change thresold with result
                if distances[index_min] < 0.5:
                    genlabel = nSpeakerVoices[index_min]["name"]
                    if genlabel not in new_labels:
                        new_labels.append(genlabel)
                    for span in audio_spans:
                        if span["label"] == label:
                            span.update({"label": genlabel})

        blocks = [{"view_id": "audio_manual"}]
        new_labels = sorted(new_labels)
        all_labels = new_labels + labels
        for label in all_labels:
            blocks.append(
                {
                    "view_id": "text_input",
                    "field_id": label,
                    "field_label": label,
                    "field_rows": 1,
                    "field_placeholder": label,
                    "field_suggestions": list(nSpeakerVoices["name"]),
                }
            )

        yield {
            "path": path,
            "text": text,
            "audio": audio,
            "audio_spans": audio_spans,
            "audio_spans_original": deepcopy(audio_spans),
            "chunk": {"start": excerpt.start, "end": excerpt.end},
            "config": {"labels": all_labels, "blocks": blocks},
            "meta": {
                "file": filename,
                "start": f"{excerpt.start:.1f}",
                "end": f"{excerpt.end:.1f}",
            },
        }


@prodigy.recipe(
    "pyannote.diarization",
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
    embeddings=(
        "Path to already created embeddings in a structured np.array with dtype=[('name', 'U100'), ('embedding', 'f4', dim), ('nb','i4')]",
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
)
def diarization(
    dataset: str,
    source: Path,
    pipeline: Union[str, Iterable[dict]],
    chunk: float = 30.0,
    num_classes: int = 4,
    embeddings: str = "",
    precision: int = 200,
    beep: bool = False,
) -> Dict[str, Any]:

    global nSpeakerVoices
    pipeline = Pipeline.from_pretrained(pipeline)
    classes = pipeline.classes()

    if isinstance(classes, Iterator):
        labels = [x for _, x in zip(range(num_classes), classes)]
    else:
        labels = classes

    if embeddings != "":
        nSpeakerVoices = np.load(embeddings)

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

    hstream = (
        set_hashes(eg, input_keys=("path", "chunk"))
        for eg in diarization_stream(
            pipeline, source, labels, chunk=chunk, randomize=False
        )
    )

    # USE "instant_submit": True in prodigy.json
    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": hstream,
        "before_db": before_db,
        "validate_answer": validate_answer,
        "update": update,
        "on_exit": on_exit,
        "config": {
            "exclude_by": "input",
            "javascript": javascript,
            "instructions": instructions_html,
            "blocks": [
                {"view_id": "audio_manual"},
                {"view_id": "text_input", "field_id": "user_input_a", "field_rows": 1},
            ],
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
