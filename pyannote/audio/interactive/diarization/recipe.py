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
import hashlib
from collections.abc import Iterator
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Iterable, Union

import numpy as np
import prodigy
from prodigy import set_hashes

from pyannote.audio import Pipeline

from ..common.utils import before_db_diarization
from .recipehelper import RecipeHelper


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
        "option",
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
    qwerty=("QWERTY keyboard", "flag", None, bool),
)
def diarization(
    dataset: str,
    source: Path,
    pipeline: Union[str, Iterable[dict]] = "pyannote/speaker-segmentation",
    chunk: float = 20.0,
    num_classes: int = 4,
    precision: int = 200,
    beep: bool = False,
    qwerty: bool = False,
) -> Dict[str, Any]:

    pipeline = Pipeline.from_pretrained(pipeline)
    classes = pipeline.classes()

    if isinstance(classes, Iterator):
        labels = [x for _, x in zip(range(num_classes), classes)]
    else:
        labels = classes

    prodigy_path = prodigy.components.db.get_prodigy_path()
    hashed_source = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    speakers_dir = Path(prodigy_path + "/" + hashed_source)
    helper = RecipeHelper(chunk, speakers_dir)

    if speakers_dir.exists():
        npy = speakers_dir / "speakers.npy"
        if npy.exists():
            helper.speakers = np.load(npy)
    else:
        speakers_dir.mkdir()

    recipe_dir = Path(__file__).resolve().parent
    common_dir = recipe_dir.parent / "common"
    controller_js = common_dir / "controller.js"
    controllerDiarization = recipe_dir / "controller.js"

    with open(controller_js) as c, open(controllerDiarization) as c_dia:
        javascript = c.read() + "\n" + c_dia.read()

    # TODO: improve this part
    template = common_dir / "instructionsDiarization.html"
    png = common_dir / "commandsDiarization.png"
    _, instructions_html = mkstemp(text=True)
    with open(instructions_html, "w") as instructions_f, open(
        template, "r"
    ) as fp_tpl, open(png, "rb") as fp_png:
        b64 = base64.b64encode(fp_png.read()).decode("utf-8")
        instructions_f.write(fp_tpl.read().replace("{IMAGE}", b64))

    hashed_stream = (
        set_hashes(eg, input_keys=("path", "chunk"), ignore=[])
        for eg in helper.stream(pipeline, source, labels, randomize=False)
    )

    left = "a" if qwerty else "q"

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": hashed_stream,
        "before_db": before_db_diarization,
        "validate_answer": helper.validate_answer,
        "update": helper.update,
        "on_exit": helper.on_exit,
        "config": {
            "exclude_by": "input",
            "instant_submit": True,
            "batch_size": 1,
            "javascript": javascript,
            "instructions": instructions_html,
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
                    ]
                }
            },
            "blocks": [
                {"view_id": "audio_manual"},
                {"view_id": "text_input", "field_id": "user_input_a", "field_rows": 1},
            ],
            "buttons": ["accept", "ignore", "undo"],
            "keymap": {
                "accept": ["enter"],
                "ignore": ["escape"],
                "reset": ["u"],
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
            "left": left,
            "right": "d",
        },
    }
