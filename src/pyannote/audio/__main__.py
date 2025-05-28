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

import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pyannote.database
import torch
import typer
import yaml
from pyannote.core import Annotation
from pyannote.pipeline.optimizer import Optimizer
from typing_extensions import Annotated

from pyannote.audio import Audio, Pipeline
from pyannote.audio.utils.postprocessing import MinDurationOffOptimizer
from pyannote.metrics.diarization import DiarizationErrorRate


class Subset(str, Enum):
    train = "train"
    development = "development"
    test = "test"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class NumSpeakers(str, Enum):
    ORACLE = "oracle"
    AUTO = "auto"


def parse_device(device: Device) -> torch.device:
    if device == Device.AUTO:
        if torch.cuda.is_available():
            device = Device.CUDA

        elif torch.backends.mps.is_available():
            device = Device.MPS

        else:
            device = Device.CPU

    return torch.device(device.value)


app = typer.Typer()


@app.command("optimize")
def optimize(
    pipeline: Annotated[
        Path,
        typer.Argument(
            help="Path to pipeline YAML configuration file",
            exists=True,
            dir_okay=False,
            file_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ],
    protocol: Annotated[
        str,
        typer.Argument(help="Protocol used for optimization"),
    ],
    subset: Annotated[
        Subset,
        typer.Option(
            help="Subset used for optimization",
            case_sensitive=False,
        ),
    ] = Subset.development,
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
    max_iterations: Annotated[
        Optional[int],
        typer.Option(help="Number of iterations to run. Defaults to run indefinitely."),
    ] = None,
    num_speakers: Annotated[
        NumSpeakers, typer.Option(help="Number of speakers (oracle or auto)")
    ] = NumSpeakers.AUTO,
):
    """
    Optimize a PIPELINE
    """

    # load pipeline configuration file in memory. this will
    # be dumped later to disk with optimized parameters
    with open(pipeline, "r") as fp:
        original_config = yaml.load(fp, Loader=yaml.SafeLoader)

    # load pipeline
    optimized_pipeline = Pipeline.from_pretrained(pipeline)
    if optimized_pipeline is None:
        raise ValueError(f"Could not load pipeline from {pipeline}")

    # send pipeline to device
    torch_device = parse_device(device)
    optimized_pipeline.to(torch_device)

    # load protocol from (optional) registry
    if registry:
        pyannote.database.registry.load_database(registry)

    preprocessors = {"audio": pyannote.database.FileFinder()}

    # pass number of speakers to pipeline if requested
    if num_speakers == NumSpeakers.ORACLE:
        preprocessors["pipeline_kwargs"] = lambda protocol_file: {
            "num_speakers": len(protocol_file["annotation"].labels())
        }

    loaded_protocol = pyannote.database.registry.get_protocol(
        protocol, preprocessors=preprocessors
    )

    files: list[pyannote.database.ProtocolFile] = list(
        getattr(loaded_protocol, subset.value)()
    )

    # setting study name to this allows to store multiple optimizations
    # for the same pipeline in the same database
    study_name = f"{protocol}.{subset.value}"
    # add suffix if we are using oracle number of speakers
    if num_speakers == NumSpeakers.ORACLE:
        study_name += ".OracleNumSpeakers"

    # journal file to store optimization results
    # if pipeline path is "config.yml", it will be stored in "config.journal"
    journal = pipeline.with_suffix(".journal")

    result: Path = pipeline.with_suffix(f".{study_name}.yaml")

    optimizer = Optimizer(
        optimized_pipeline,
        db=journal,
        study_name=study_name,
        sampler=None,  # TODO: support sampler
        pruner=None,  # TODO: support pruner
        average_case=False,
    )

    direction = 1 if optimized_pipeline.get_direction() == "minimize" else -1

    # read best loss so far
    global_best_loss: float = optimizer.best_loss
    local_best_loss: float = global_best_loss

    #
    try:
        warm_start = optimized_pipeline.default_parameters()
    except NotImplementedError:
        warm_start = None

    iterations = optimizer.tune_iter(files, warm_start=warm_start)

    # TODO: use pipeline.default_params() as warm_start?

    for i, status in enumerate(iterations):
        loss = status["loss"]

        # check whether this iteration led to a better loss
        # than all previous iterations for this run
        if direction * (loss - local_best_loss) < 0:
            # new (local) best loss
            local_best_loss = loss

            # if it did, also check directly from the central database if this is a new global best loss
            # (we might have multiple optimizations going on simultaneously)
            if local_best_loss == (global_best_loss := optimizer.best_loss):
                # if we have a new global best loss, save it to disk
                original_config["params"] = status["params"]
                original_config["optimization"] = {
                    "protocol": protocol,
                    "subset": subset.value,
                    "status": {
                        "best_loss": local_best_loss,
                        "last_updated": datetime.now().isoformat(),
                    },
                }
                with open(result, "w") as fp:
                    yaml.dump(original_config, fp)

            local_best_loss = global_best_loss

        if max_iterations and i + 1 >= max_iterations:
            break


@app.command("download")
def download(
    pipeline: Annotated[
        str,
        typer.Argument(
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-3.1)"
        ),
    ],
    token: Annotated[
        str,
        typer.Argument(
            help="Huggingface token to be used for downloading from Huggingface hub."
        ),
    ],
    cache: Annotated[
        Path,
        typer.Option(
            help="Path to the folder where files downloaded from Huggingface hub are stored.",
            exists=True,
            dir_okay=True,
            file_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
):
    """
    Download a pretrained PIPELINE to disk for later offline use.
    """

    # load pretrained pipeline
    _ = Pipeline.from_pretrained(pipeline, token=token, cache_dir=cache)


@app.command("apply")
def apply(
    pipeline: Annotated[
        str,
        typer.Argument(
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-3.1)"
        ),
    ],
    audio: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file",
            exists=True,
            file_okay=True,
            readable=True,
        ),
    ],
    into: Annotated[
        Path,
        typer.Option(
            help="Path to file where results are saved.",
            exists=False,
            dir_okay=False,
            file_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    device: Annotated[
        Device, typer.Option(help="Accelerator to use (CPU, CUDA, MPS)")
    ] = Device.AUTO,
    cache: Annotated[
        Path,
        typer.Option(
            help="Path to the folder where files downloaded from Huggingface hub are stored.",
            exists=True,
            dir_okay=True,
            file_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
):
    """
    Apply a pretrained PIPELINE to an AUDIO file
    """

    # load pretrained pipeline
    pretrained_pipeline = Pipeline.from_pretrained(pipeline, cache_dir=cache)

    # send pipeline to device
    torch_device = parse_device(device)
    pretrained_pipeline.to(torch_device)

    # apply pipeline to audio file
    prediction: Annotation = pretrained_pipeline(audio)

    # save (or print) results
    with open(into, "w") if into else nullcontext(sys.stdout) as rttm:
        prediction.write_rttm(rttm)


@app.command("benchmark")
def benchmark(
    pipeline: Annotated[
        str,
        typer.Argument(
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-3.1)"
        ),
    ],
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
    num_speakers: Annotated[
        NumSpeakers, typer.Option(help="Number of speakers (oracle or auto)")
    ] = NumSpeakers.AUTO,
    cache: Annotated[
        Path,
        typer.Option(
            help="Path to the folder where files downloaded from Huggingface hub are stored.",
            exists=True,
            dir_okay=True,
            file_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    postprocess: Annotated[
        bool,
        typer.Option(
            help="Evaluate both original and post-processed predictions.",
        )
    ] = False,
):
    """
    Benchmark a pretrained diarization PIPELINE

    This will run the pipeline on all files in the specified protocol and subset,
    save the results in RTTM format, and compute the Diarization Error Rate (DER)
    for each file. If `--postprocess` is used, it will also postprocess predictions
    by filling short within speaker gaps and save the results in a separate file.
    """

    # load pretrained pipeline
    pretrained_pipeline = Pipeline.from_pretrained(pipeline, cache_dir=cache)
    if pretrained_pipeline is None:
        print(f"Could not load pretrained pipeline from {pipeline}.")
        raise typer.exit(code=1)

    # send pipeline to device
    torch_device = parse_device(device)
    pretrained_pipeline.to(torch_device)

    # load protocol from (optional) registry
    if registry:
        pyannote.database.registry.load_database(registry)

    preprocessors = {"audio": pyannote.database.FileFinder()}

    # pass number of speakers to pipeline if requested
    if num_speakers == NumSpeakers.ORACLE:
        preprocessors["pipeline_kwargs"] = lambda protocol_file: {
            "num_speakers": len(protocol_file["annotation"].labels())
        }

    loaded_protocol = pyannote.database.registry.get_protocol(
        protocol, preprocessors=preprocessors
    )

    benchmark_name = f"{protocol}.{subset.value}"
    if num_speakers == NumSpeakers.ORACLE:
        benchmark_name += ".OracleNumSpeakers"

    processing_time: dict[str, float] = dict()
    playing_time: dict[str, float] = dict()

    metric = DiarizationErrorRate()

    serialized_predictions: dict[str, dict] = dict()

    with open(into / f"{benchmark_name}.rttm", "w") as rttm:

        files = list(getattr(loaded_protocol, subset.value)())
        for file in files:

            uri: str = file["uri"]
            playing_time[uri] = Audio().get_duration(file)

            tic: float = time.time()

            prediction = pretrained_pipeline(file, **file.get("pipeline_kwargs", {}))

            tac: float = time.time()
            processing_time[uri] = tac - tic

            # if prediction has a built-in serialize method, save serialized version
            if hasattr(prediction, "serialize"):
                serialized_predictions[uri] = prediction.serialize()

            # if result is an Annotation, assume it is speaker diarization
            if isinstance(prediction, Annotation):
                pass

            # if result contains a dedicated output for diarization, use it
            elif hasattr(prediction, "speaker_diarization"):
                prediction: Annotation = prediction.speaker_diarization

            # raise an error if no speaker diarization is found
            else:
                raise ValueError("Could not find speaker diarization in results.")

            prediction.write_rttm(rttm)
            rttm.flush()

            annotation = file.get("annotation", None)
            if annotation is None:
                continue

            annotated = file.get("annotated", None)

            _ = metric(
                prediction,
                annotation,
                uem=annotated,
            )

            # cache predictions
            file["speaker_diarization"] = prediction


    # save serialized predictions to disk (might contain more than just diarization results)
    if serialized_predictions:
        with open(into / f"{benchmark_name}.json", "w") as f:
            json.dump(serialized_predictions, f, indent=2)

    # save metric results in both CSV and human-readable formats 
    with open(into / f"{benchmark_name}.csv", "w") as csv:
        metric.report().to_csv(csv)

    with open(into / f"{benchmark_name}.txt", "w") as txt:
        txt.write(str(metric))

    # log processing time and capacity
    processing = dict()
    total_processing_time: float = sum(processing_time.values())
    total_playing_time: float = sum(playing_time.values())
    processing["seconds_per_hour"] = total_processing_time / (total_playing_time / 3600)
    processing["times_faster_than_realtime"] = (
        total_playing_time / total_processing_time
    )
    processing["total_processing_time"] = total_processing_time

    if torch_device.type == "cuda":

        props = torch.cuda.get_device_properties(torch_device)
        props_dict = {}
        for attr in dir(props):
            if not attr.startswith("_"):
                value = getattr(props, attr)
                # Only include basic types (skip unpicklable like _CUuuid)
                if isinstance(value, (int, float, str, bool, tuple, list)):
                    props_dict[attr] = value

        processing["device"] = props_dict
        device_name = props_dict["name"].replace(" ", "-")
        speed_yml = into / f"{benchmark_name}.{device_name}.yml"

    else:
        speed_yml = into / f"{benchmark_name}.yml"

    # report metric results with an optimized min_duration_off
    if postprocess:

        minDurationOffOptimizer = MinDurationOffOptimizer()
        best_min_duration_off, best_report = minDurationOffOptimizer(files, metric)

        with open(into / f"{benchmark_name}.Optimized.csv", "w") as csv:
            best_report.to_csv(csv)
        with open(into / f"{benchmark_name}.Optimized.txt", "w") as txt:
            txt.write(str(best_report))

        # report collar best value
        with open(into / f"{benchmark_name}.Optimized.yml", "w") as yml:
            yaml.dump({"min_duration_off": best_min_duration_off}, yml)


    with open(speed_yml, "w") as yml:
        yaml.dump(processing, yml)


if __name__ == "__main__":
    app()
