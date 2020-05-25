# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHORS
# Hervé Bredin - http://herve.niderb.fr

from pathlib import Path
from typing import Text, Dict
import prodigy
from .pipeline import InteractiveDiarization

PRETRAINED_PARAMS = {
    "emb_duration": 1.7657045140297274,
    "emb_step_ratio": 0.20414598809353782,
    "emb_threshold": 0.5274911675340328,
    "sad_min_duration_off": 0.13583405625051126,
    "sad_min_duration_on": 0.0014190874731107286,
    "sad_threshold_off": 0.7878607185085043,
    "sad_threshold_on": 0.5940560764213958,
}


@prodigy.recipe(
    "pyannote.sad.manual",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Directory containing audio files to annotate", "positional", None, Path),
    chunk=(
        "Split long audio files into shorter chunks of {chunk} seconds each",
        "option",
        None,
        float,
    ),
)
def sad_manual(dataset: Text, source: Path, chunk: float = 30.0) -> Dict:

    pipeline = InteractiveDiarization().instantiate(PRETRAINED_PARAMS)

    return {
        "dataset": dataset,
        "view_id": "audio_manual",
        "stream": pipeline.prodigy_sad_manual_stream(source, chunk=chunk),
        "before_db": pipeline.prodigy_sad_manual_before_db,
        "config": {
            "audio_autoplay": True,
            "audio_loop": True,
            "show_audio_minimap": False,
            "audio_bar_width": 3,
            "audio_bar_height": 1,
            "labels": ["SPEECH",],
        },
    }


@prodigy.recipe(
    "pyannote.dia.binary",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Directory containing audio files to annotate", "positional", None, Path),
)
def dia_binary(dataset: Text, source: Path) -> Dict:

    pipeline = InteractiveDiarization().instantiate(PRETRAINED_PARAMS)

    return {
        "dataset": dataset,
        "view_id": "audio",
        "stream": pipeline.prodigy_dia_binary_stream(dataset, source),
        "update": pipeline.prodigy_dia_binary_update,
        "before_db": pipeline.prodigy_dia_binary_before_db,
        "config": {
            "audio_autoplay": True,
            "audio_loop": True,
            "show_audio_minimap": False,
            "audio_bar_width": 3,
            "audio_bar_height": 1,
            "labels": ["SPEAKER", "SAME_SPEAKER"],
            "batch_size": 1,
            "instant_submit": True,
        },
    }
