#!/usr/bin/env python
# encoding: utf-8

# MIT License
#
# Copyright (c) 2024- CNRS
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


import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import pyannote.database
import torch
import typer
from pyannote.core import Annotation
from typing_extensions import Annotated

from pyannote.audio import Pipeline


class Subset(str, Enum):
    train = "train"
    development = "development"
    test = "test"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


def guess_device() -> Device:
    if torch.cuda.is_available():
        return Device.CUDA

    if torch.backends.mps.is_available():
        return Device.MPS

    return Device.CPU


app = typer.Typer()


@app.command("apply")
def apply(
    audio: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file",
            exists=True,
            file_okay=True,
            readable=True,
        ),
    ],
    pipeline: Annotated[
        str, typer.Option(help="Pretrained pipeline")
    ] = "pyannote/speaker-diarization-3.1",
    device: Annotated[
        Device, typer.Option(help="Accelerator to use (CPU, CUDA, MPS)")
    ] = Device.AUTO,
):
    # load pretrained pipeline
    pretrained_pipeline = Pipeline.from_pretrained(pipeline)

    # send pipeline to device
    if device == Device.AUTO:
        device = guess_device()
    torch_device = torch.device(device.value)
    pretrained_pipeline.to(torch_device)

    prediction: Annotation = pretrained_pipeline(audio)

    prediction.write_rttm(sys.stdout)
    sys.stdout.flush()


@app.command("benchmark")
def benchmark(
    protocol: Annotated[
        str,
        typer.Argument(help="Benchmarked protocol"),
    ],
    into: Annotated[
        Path,
        typer.Argument(
            help="Directory into which benchmark results are saved",
            exists=True,
            dir_okay=True,
            file_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ],
    subset: Annotated[
        Subset,
        typer.Option(
            help="Benchmarked subset",
            case_sensitive=False,
        ),
    ] = Subset.test,
    pipeline: Annotated[
        str, typer.Option(help="Benchmarked pipeline")
    ] = "pyannote/speaker-diarization-3.1",
    device: Annotated[
        Device, typer.Option(help="Accelerator to use (CPU, CUDA, MPS)")
    ] = Device.AUTO,
    registry: Annotated[
        Optional[Path],
        typer.Option(
            help="Loaded registry",
            exists=True,
            dir_okay=False,
            file_okay=True,
            readable=True,
        ),
    ] = None,
):
    # load pretrained pipeline
    pretrained_pipeline = Pipeline.from_pretrained(pipeline)

    # send pipeline to device
    if device == Device.AUTO:
        device = guess_device()
    torch_device = torch.device(device.value)
    pretrained_pipeline.to(torch_device)

    # load pipeline metric (when available)
    try:
        metric = pretrained_pipeline.get_metric()
    except NotImplementedError:
        metric = None

    # load protocol from (optional) registry
    if registry:
        pyannote.database.registry.load_database(registry)

    loaded_protocol = pyannote.database.registry.get_protocol(
        protocol, {"audio": pyannote.database.FileFinder()}
    )

    with open(into / f"{protocol}.{subset}.rttm", "w") as rttm:
        for file in getattr(loaded_protocol, subset.value)():
            prediction: Annotation = pretrained_pipeline(file)
            prediction.write_rttm(rttm)
            rttm.flush()

            if metric is None:
                continue

            groundtruth = file.get("annotation", None)
            if groundtruth is None:
                continue

            annotated = file.get("annotated", None)
            _ = metric(groundtruth, prediction, uem=annotated)

    if metric is None:
        return

    with open(into / f"{protocol}.{subset}.txt", "w") as txt:
        txt.write(str(metric))

    print(str(metric))


if __name__ == "__main__":
    app()
