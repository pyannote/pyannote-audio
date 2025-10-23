#!/usr/bin/env python
# encoding: utf-8

# MIT License
#
# Copyright (c) 2024-2025 CNRS
# Copyright (c) 2025 pyannoteAI
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
import types
from contextlib import nullcontext
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pyannote.database
import torch
import typer
import yaml
from pyannote.audio import Audio, Pipeline, Model
from pyannote.core import Annotation
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.pipeline.optimizer import Optimizer
from rich.progress import track
from scipy.optimize import minimize_scalar
from typing_extensions import Annotated


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


class Metric(str, Enum):
    DiarizationErrorRate = "DiarizationErrorRate"
    JaccardErrorRate = "JaccardErrorRate"

    @classmethod
    def from_str(cls, metric: str):
        """Convert a string to a Metric enum value."""

        if metric == "DiarizationErrorRate":
            return DiarizationErrorRate()
        elif metric == "JaccardErrorRate":
            return JaccardErrorRate()


def parse_device(device: Device) -> torch.device:
    if device == Device.AUTO:
        if torch.cuda.is_available():
            device = Device.CUDA

        elif torch.backends.mps.is_available():
            device = Device.MPS

        else:
            device = Device.CPU

    return torch.device(device.value)


def get_diarization(prediction) -> Annotation:
    # if result is an Annotation, assume it is speaker diarization
    if isinstance(prediction, Annotation):
        return prediction

    # if result contains a dedicated output for diarization, use it
    if hasattr(prediction, "speaker_diarization"):
        return prediction.speaker_diarization

    raise ValueError("Could not find speaker diarization in prediction.")


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
    metric: Annotated[
        Metric,
        typer.Option(
            help="Metric to optimize against",
            case_sensitive=False,
        ),
    ] = Metric.DiarizationErrorRate,
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
        print(f"Could not load pipeline from {pipeline}.")
        raise typer.exit(code=1)

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

    # update `get_metric` method to return the requested metric instance
    def _get_metric(self):
        return Metric.from_str(metric)

    optimized_pipeline.get_metric = types.MethodType(_get_metric, optimized_pipeline)

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
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-community-1)"
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
    pretrained_pipeline = Pipeline.from_pretrained(
        pipeline, token=token, cache_dir=cache
    )
    if pretrained_pipeline is None:
        print(f"Could not load pretrained pipeline from {pipeline}.")
        raise typer.exit(code=1)


@app.command("apply")
def apply(
    pipeline: Annotated[
        str,
        typer.Argument(
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-community-1)"
        ),
    ],
    audio: Annotated[
        Path,
        typer.Argument(
            help="Path to audio file or directory",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ],
    into: Annotated[
        Path,
        typer.Option(
            help="Path to file or directory where results are saved.",
            exists=False,
            dir_okay=True,
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
    Apply a pretrained PIPELINE to an AUDIO file or directory
    """

    # load pretrained pipeline
    pretrained_pipeline = Pipeline.from_pretrained(pipeline, cache_dir=cache)
    if pretrained_pipeline is None:
        print(f"Could not load pretrained pipeline from {pipeline}.")
        raise typer.exit(code=1)

    # send pipeline to device
    torch_device = parse_device(device)
    pretrained_pipeline.to(torch_device)

    if audio.is_dir():

        if into is None or not into.is_dir():
            typer.echo("When AUDIO is a directory, INTO must also be a directory.")
            raise typer.exit(code=1)

        inputs: list[Path] = sorted(path for path in audio.iterdir() if path.is_file())
        rttms: list[Path | None] = [into / (path.stem + ".rttm") for path in inputs]
        jsons: list[Path | None] = [into / (path.stem + ".json") for path in inputs]

    else:
        
        if not (into is None or into.is_file()):
            typer.echo("When AUDIO is a file, INTO must also be a file.")
            raise typer.exit(code=1)

        inputs = [audio]
        rttms: list[Path | None] = [into]
        jsons: list[Path | None] = [into.with_suffix(".json") if into else None]

    for current_input, current_rttm, current_json in zip(inputs, rttms, jsons):

        prediction = pretrained_pipeline(current_input)

        speaker_diarization = get_diarization(prediction)

        with open(current_rttm, "w") if current_rttm else nullcontext(sys.stdout) as r:
            speaker_diarization.write_rttm(r)

        if hasattr(prediction, "serialize") and current_json:
            serialized: dict = prediction.serialize()
            with open(current_json, "w") as j:
                json.dump(serialized, j, indent=2)


class MinDurationOffOptimizer:
    """Utility to optimize `min_duration_off`

    Depending on the pipeline used for speaker diarization, short breaks within speaker turns
    (e.g. between each word) might lead to unfair missed detection rates.

    This utility aims at finding the best value for `min_duration_off` parameter that controls
    how short a within-speaker gap must be to be filled.
    """

    def _compute_metric(self, files, metric, collar: float) -> float:
        metric.reset()
        for file in files:
            file["temporary_speaker_diarization"] = file["speaker_diarization"].support(
                collar=collar
            )
            _ = metric(
                file["annotation"],
                file["temporary_speaker_diarization"],
                uem=file["annotated"],
            )
        self._reports[collar] = metric.report()

        current_metric = abs(metric)

        # store best postprocessed speaker diarization
        if current_metric < self._best_metric:
            self._best_metric = current_metric
            for file in files:
                file["best_speaker_diarization"] = file.pop(
                    "temporary_speaker_diarization"
                )

        return current_metric

    def __call__(
        self, files, metric: BaseMetric, bounds: tuple[float, float] = (0.0, 1.0)
    ) -> tuple[float, "DataFrame"]:
        """Optimize 'min_duration_off' value for `metric`

        Parameters
        ----------
        files : list[dict]
            List of dictionaries containing 'annotation', 'annotated',
            and 'speaker_diarization' keys.
        metric : BaseMetric
            Metric to optimize against (usually a DiarizationErrorRate instance).
        bounds : tuple[float, float], optional
            Lower and upper bounds for the `min_duration_off` parameter (in seconds).
            Defaults to (0.0, 1.0).

        Returns
        -------
        best_min_duration_off : float
            Optimized min_duration_off parameter.
        best_report: pandas.DataFrame
            Corresponding pyannote.metrics report.
        """

        # assumes metric should be minimized
        # TODO: use -inf when/if metric should be maximized
        self._best_metric = float("inf")
        self._reports: dict[float, "DataFrame"] = dict()

        # force test with no collar
        no_collar_metric = self._compute_metric(files, metric, 0.0)

        res = minimize_scalar(
            partial(self._compute_metric, files, metric),
            bounds=bounds,
            method="Bounded",
        )

        # in case where better results are obtained without a collar
        if no_collar_metric == self._best_metric:
            best_min_duration_off = 0.0

        else:
            best_min_duration_off = float(res.x)

        return best_min_duration_off, self._reports[best_min_duration_off]


@app.command("benchmark")
def benchmark(
    pipeline: Annotated[
        str,
        typer.Argument(
            help="Pretrained pipeline (e.g. pyannote/speaker-diarization-community-1)"
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
    optimize: Annotated[
        bool,
        typer.Option(
            help="Evaluate both original and post-processed predictions.",
        ),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option(
            help="Show progress",
        ),
    ] = False,
    per_file: Annotated[
        bool,
        typer.Option(
            help="Save one RTTM/JSON file per processed audio file."
        )
    ] = False,
):
    """
    Benchmark a pretrained diarization PIPELINE

    This will run the pipeline on all files in the specified protocol and subset,
    save the results in RTTM format, and compute the Diarization Error Rate (DER)
    for each file. If `--optimize` is used, it will also post-process predictions
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

    # make sure an "audio" key is available in protocol files
    preprocessors = {"audio": pyannote.database.FileFinder()}

    # pass number of speakers to pipeline if requested
    if num_speakers == NumSpeakers.ORACLE:
        preprocessors["pipeline_kwargs"] = lambda protocol_file: {
            "num_speakers": len(protocol_file["annotation"].labels())
        }

    # load benchmark files
    loaded_protocol = pyannote.database.registry.get_protocol(
        protocol, preprocessors=preprocessors
    )
    files = list(getattr(loaded_protocol, subset.value)())

    # check that manual annotation is available for all files
    # (condition to actually run the benchmark)
    skip_metric = False
    if any(file.get("annotation", None) is None for file in files):
        print(
            f"Manual annotation is not available for files in {protocol} {subset.value} subset so skipping metric evaluation."
        )
        skip_metric = True

    # `benchmark_name` is used as prefix to output files
    benchmark_name = f"{protocol}.{subset.value}"
    if num_speakers == NumSpeakers.ORACLE:
        benchmark_name += ".OracleNumSpeakers"

    # used to store processing time and file duration
    processing_time: dict[str, float] = dict()
    playing_time: dict[str, float] = dict()

    # used to store raw predictions in JSON format
    serialized_predictions: dict[str, dict] = dict()

    if not skip_metric:
        # initialize diarization error rate metric
        metric = DiarizationErrorRate()

    # speaker count confusion matrix
    # speaker_count[i][j] is the number of files with i speakers in the
    # manual annotation and j speakers in the prediction
    speaker_count: dict[int, dict[int, int]] = dict()

    if per_file:
        benchmark_dir = into / benchmark_name
        if benchmark_dir.exists():
            raise FileExistsError(f"{benchmark_dir} already exists.")

        rttm_dir = benchmark_dir / "rttm"
        rttm_dir.mkdir(parents=True)

    else:
        rttm_file = into / f"{benchmark_name}.rttm"
        # make sure we don't overwrite previous results
        if rttm_file.exists():
            raise FileExistsError(f"{rttm_file} already exists.")

    # iterate over all files in the specified subset
    for file in track(files, disable=not progress):
        # gather file metadata
        uri: str = file["uri"]
        playing_time[uri] = Audio().get_duration(file)

        tic: float = time.time()

        # apply pretrained pipeline to file
        prediction = pretrained_pipeline(file, **file.get("pipeline_kwargs", {}))

        tac: float = time.time()
        processing_time[uri] = tac - tic

        # if prediction has a built-in serialize method, save serialized version
        if hasattr(prediction, "serialize"):
            if per_file:
                json_dir = benchmark_dir / "json"
                json_dir.mkdir(exist_ok=True)

                with open(json_dir / f"{uri}.json", "w") as f:
                    json.dump(prediction.serialize(), f, indent=2)
            else:
                serialized_predictions[uri] = prediction.serialize()

        # get speaker diarization from raw prediction
        speaker_diarization = get_diarization(prediction)

        # dump prediction to RTTM file
        if per_file:
            rttm_file = rttm_dir / f"{uri}.rttm"

        with open(rttm_file, "w" if per_file else "a") as rttm:
            speaker_diarization.write_rttm(rttm)

        # compute metric when possible
        if not skip_metric:
            _ = metric(
                file["annotation"],
                speaker_diarization,
                uem=file.get("annotated", None),
            )

        # increment speaker count confusion matrix
        pred_num_speakers: int = len(speaker_diarization.labels())
        true_num_speakers: int = len(file["annotation"].labels())
        speaker_count.setdefault(true_num_speakers, dict()).setdefault(
            pred_num_speakers, 0
        )
        speaker_count[true_num_speakers][pred_num_speakers] += 1

        # keep track of prediction for later "min_duration_off" optimization
        if optimize:
            file["speaker_diarization"] = speaker_diarization

    # save serialized predictions to disk (might contain more than just diarization results)
    if serialized_predictions and not per_file:
        with open(into / f"{benchmark_name}.json", "w") as f:
            json.dump(serialized_predictions, f, indent=2)

    # log processing time and capacity
    processing = dict()
    total_processing_time: float = sum(processing_time.values())
    total_playing_time: float = sum(playing_time.values())
    processing["seconds_per_hour"] = total_processing_time / (total_playing_time / 3600)
    processing["times_faster_than_realtime"] = (
        total_playing_time / total_processing_time
    )
    processing["total_processing_time"] = total_processing_time

    # keep track of GPU device properties
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

    with open(speed_yml, "w") as yml:
        yaml.dump(processing, yml)

    # no need to go further than this point if evaluation is not possible
    if skip_metric:
        raise typer.exit()

    # save metric results in both CSV and human-readable formats
    with open(into / f"{benchmark_name}.csv", "w") as csv:
        metric.report().to_csv(csv)

    with open(into / f"{benchmark_name}.txt", "w") as txt:
        txt.write(str(metric))

    # turn speaker count confusion matrix into numpy array
    # and save it to disk as a CSV file
    max_true_speakers = max(speaker_count.keys())
    max_pred_speakers = max(
        max(speaker_count[true_speakers].keys())
        for true_speakers in speaker_count.keys()
    )
    speaker_count_matrix = np.zeros(
        (max_true_speakers + 1, max_pred_speakers + 1), dtype=int
    )
    for true_speakers, pred_counts in speaker_count.items():
        for pred_speakers, count in pred_counts.items():
            speaker_count_matrix[true_speakers, pred_speakers] = count

    # compute the average error in the speaker count prediction
    speaker_count_error: float = np.sum(
        [
            abs(true_speakers - pred_speakers) * count
            for true_speakers, pred_counts in speaker_count.items()
            for pred_speakers, count in pred_counts.items()
        ]
    ) / np.sum(speaker_count_matrix)

    # compute the accuracy of the speaker count prediction
    speaker_count_accuracy: float = np.sum(np.diag(speaker_count_matrix)) / np.sum(
        speaker_count_matrix
    )

    np.savetxt(
        into / f"{benchmark_name}.SpeakerCount.csv",
        speaker_count_matrix,
        delimiter=",",
        fmt="%3d",
        footer=f"Accuracy = {speaker_count_accuracy:.1%} / Average error = {speaker_count_error:.2f} speakers off",
    )

    # report metric results with an optimized min_duration_off
    if optimize:
        minDurationOffOptimizer = MinDurationOffOptimizer()
        best_min_duration_off, best_report = minDurationOffOptimizer(files, metric)

        with open(into / f"{benchmark_name}.OptimizedMinDurationOff.csv", "w") as csv:
            best_report.to_csv(csv)

        with open(into / f"{benchmark_name}.OptimizedMinDurationOff.txt", "w") as txt:
            txt.write(
                best_report.to_string(
                    sparsify=False, float_format=lambda f: "{0:.2f}".format(f)
                )
            )

        # keep track of the best `min_duration_off` value for later reference
        with open(into / f"{benchmark_name}.OptimizedMinDurationOff.yml", "w") as yml:
            yaml.dump({"min_duration_off": best_min_duration_off}, yml)

        if not per_file:
            optimized_rttm_file = into / f"{benchmark_name}.OptimizedMinDurationOff.rttm"

            # make sure we don't overwrite previous results
            if optimized_rttm_file.exists():
                raise FileExistsError(f"{optimized_rttm_file} already exists.")

        for file in files:
            if per_file:
                optimized_rttm_file = rttm_dir / f"{file['uri']}.OptimizedMinDurationOff.rttm"

            with open(optimized_rttm_file, "w" if per_file else "a") as rttm:
                file["best_speaker_diarization"].write_rttm(rttm)


@app.command("strip")
def strip(
    checkpoint: Annotated[
        Path,
        typer.Argument(
            help="Path to pyannote.audio model checkpoint",
            exists=True,
            dir_okay=False,
            file_okay=True,
            resolve_path=True,
        ),
    ],
    into: Annotated[
        Path,
        typer.Argument(
            help="Path to the stripped checkpoint",
            exists=False,
            dir_okay=False,
            file_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ],
):
    """
    Strip a pretrained CHECKPOINT to only keep the parts needed for inference.
    """

    keys = [
        "pytorch-lightning_version",   # * pytorch-lightning needs
        "hparams_name",                #   those values to initialize 
        "hyper_parameters",            #   the model architecture
        "state_dict",                  # * actual weights
        "pyannote.audio",              # * pyannote.audio dependencies 
    ]

    old_checkpoint = torch.load(
        checkpoint, map_location=torch.device("cpu"), weights_only=False
    )
    new_checkpoint = {
        key: value for key, value in old_checkpoint.items() if key in keys
    }
    torch.save(new_checkpoint, into)

    # check that the stripped checkpoint can be loaded again
    try:
        _ = Model.from_pretrained(into)
    except Exception as e:
        sys.exit(
            f"Something went wrong while stripping the checkpoint as it could not be reloaded: {e}"
        )


if __name__ == "__main__":
    app()
