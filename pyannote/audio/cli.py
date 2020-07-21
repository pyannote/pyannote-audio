#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


"""
Neural building blocks for speaker diarization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
  pyannote-audio extract                  [options] <root>     <protocol>
  pyannote-audio train    [--cpu | --gpu] [options] <root>     <protocol>
  pyannote-audio validate [--cpu | --gpu] [options] <train>    <protocol>
  pyannote-audio apply    [--cpu | --gpu] [options] <validate> <protocol>
  pyannote-audio -h | --help
  pyannote-audio --version

This command line tool can be used to train, validate, and apply neural networks
for most module of a speaker diarization pipeline

Running a complete experiment on the provided "debug" dataset would go like this:

    * Run experiment on this pyannote.database protocol
      $ export DATABASE=Debug.SpeakerDiarization.Debug

    * This directory will contain experiments artifacts:
      $ mkdir my_experiment && cd my_experiment

    * A unique configuration file describes the experiment hyper-parameters
      (see "Configuration file" below for details):
      $ edit config.yml

    * This will train the model on the training set:
      $ pyannote-audio train ${PWD} ${DATABASE}

    * Training artifacts (including model weights) are stored in a sub-directory
      whose name makes it clear which dataset and subset (train, by default)
      were used for training the model.
      $ cd train/${DATABASE}.train

    * This will validate the model on the development set:
      $ pyannote-audio validate ${PWD} ${DATABASE}

    * Validation artifacts (including the selection of the best epoch) are
      stored in a sub-directory named after the dataset and subset (development,
      by default) used for validating the model.
      $ cd validate/${DATABASE}.development

    * This will apply the best model (according to the validation step) to the
      test set:
      $ pyannote-audio apply ${PWD} ${DATABASE}

    * Inference artifacts are stored in a sub-directory whose name makes it
      clear which epoch has been used (e.g. apply/0125). Artifacts include:
        * raw output of the best model (one numpy array per file  than can be
          loaded with pyannote.audio.features.Precomputed API and handled with
          pyannote.core.SlidingWindowFeature API)
        * (depending on the task) a file "${DATABASE}.test.rttm" containing the
          post-processing of raw output.
        * (depending on the task) a file "${DATABASE}.test.eval" containing the
          evaluation result computed with pyannote.metrics.

pyannote.database support
~~~~~~~~~~~~~~~~~~~~~~~~~

PYANNOTE_DATABASE_CONFIG=

Configuration file <root>/config.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Reproducible research is facilitated by the systematic use of configuration
    files stored in <root>/config.yml in YAML format.

    .......................... <root>/config.yml ..........................
    task:
        name:
        params:

    feature_extraction:
        name:
        params:

    augmentation:
        name:
        params:

    architecture:
        name:
        params:

    scheduler:
        name:
        params:

    preprocessors:

    callbacks:
    ...................................................................

    File <root>/config.yml is mandatory, unless option --pretrained is used.

    When fine-tuning a model with option --pretrained=<model>, one can omit it
    and the original <model> configuration file is used instead. If (a possibly
    partial) <root>/config.yml file is provided anyway, it is used to override
    <model> configuration file.

Tensorboard support
~~~~~~~~~~~~~~~~~~~

    A bunch of metrics are logged during training and validation (e.g. loss,
    learning rate, computation time, validation metric). They can be visualized
    using tensorboard:

        $ tensorboard --logdir=<root>

Common options
~~~~~~~~~~~~~~

  <root>                  Experiment root directory. Should contain config.yml
                          configuration file, unless --pretrained option is
                          used (for which config.yml is optional).

  <protocol>              Name of protocol to use for training, validation, or
                          inference. Have a look at pyannote.database
                          documentation for instructions on how to define a
                          protocol with your own dataset:
                          https://github.com/pyannote/pyannote-database

  <train>                 Path to <root> sub-directory containing training
                          artifacts (e.g. <root>/train/<protocol>.train)

  <validate>              Path to <train> sub-directory containing validation
                          artifacts (e.g. <train>/validate/<protocol>.development)
                          In case option --pretrained=<model> is used, the
                          output of the pretrained model is dumped into the
                          <validate> directory.

  --subset=<subset>       Subset to use for training (resp. validation,
                          inference). Defaults to "train" (resp. "development",
                          "test") for strict enforcement of machine learning
                          good practices.

  --gpu                   Run on all available GPUs. Use CUDA_VISIBLE_DEVICES
                          environment variable to force using specific ones.

  --cpu                   Run on CPU. Defaults to using GPUs when available.

  --debug                 Run using PyTorch's anomaly detection. This will throw
                          an error if a NaN value is produced, and the stacktrace
                          will point to the origin of it. This option can
                          considerably slow execution.

  --from=<epoch>          Start training (resp. validating) at (zero-based) 
                          epoch <epoch>. Use --from=last to start from last
                          available epoch at launch time. 

  --pretrained=<model>    Warm start training with pre-trained model. Can be
                          either a path to an existing checkpoint (e.g.
                          <train>/weights/0050.pt) or the name of a model
                          available in torch.hub.list('pyannote/pyannote.audio')
                          This option can also be used to apply a pretrained
                          model. See description of <validate> for more details.

  --to=<epoch>            End training (resp. validating) at (zero-based) <epoch>.
                          Use --to=last to validate until last available epoch at 
                          launch time [default: 99]

  --every=<epoch>         Validate model every <epoch> epochs [default: 1].

  --batch=<size>          Set batch size used for validation and inference.
                          Has no effect when training as this parameter should
                          be defined in the configuration file. Defaults to
                          the one used for training.

  --duration=<duration>   Use audio chunks with that duration. Defaults to the
                          fixed duration used during training, when available.

  --step=<ratio>          Ratio of audio chunk duration used as step between
                          two consecutive audio chunks [default: 0.25]

  --parallel=<n_workers>  Use that many workers for generating training samples.
                          Defaults to half of multiprocessing.cpu_count().
"""

#   For speaker change detection, validation consists in looking for the value of
#   the peak detection threshold that maximizes the f-score of purity and
#   coverage:

#   --diarization           Use diarization purity and coverage instead of
#                           (default) segmentation purity and coverage.

#   For speaker embedding and verification protocols, validation runs the actual
#   speaker verification experiment (representing each recording by its average
#   embedding) and reports equal error rate.

#   For speaker embedding and diarization protocols, validation runs a speaker
#   diarization pipeline based on oracle segmentation and "pool-linkage"
#   agglomerative clustering of speech turns (represented by their average
#   embedding), and looks for the threshold that maximizes the f-score of purity
#   and coverage.


import pytorch_lightning as pl
pl.utilities.seed.seed_everything(42)

import yaml
import time
import warnings
from glob import glob
from tqdm import tqdm, trange
from docopt import docopt
from pathlib import Path
from typing import Text, Dict
import multiprocessing

from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.database import FileFinder
from pyannote.database import Preprocessors
from pyannote.database import Subset

from pyannote.audio import __version__
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.features.wrapper import Wrapper
from pyannote.audio.features import Pretrained
from pyannote.audio.features import RawAudio
from pyannote.audio.features import Precomputed

from pyannote.audio.train.task import BaseTask


import torch
from torch.utils.tensorboard import SummaryWriter
import zipfile
import hashlib


def load_protocol(
    protocol_name: Text, subset: Subset = "train", preprocessors: Preprocessors = None
):
    """Initialize pyannote.database protocol for use with pyannote.audio

    Automatically add "audio" and "duration" keys when they are not available

    Parameters
    ----------
    protocol_name : str
        Protocol name (e.g. Debug.SpeakerDiarization.Debug)
    subset : {"train", "development", "test"}
        Subset.
    preprocessors : Preprocessors
        Preprocessors.

    Returns
    -------
    protocol : Protocol
    """

    if preprocessors is None:
        preprocessors = dict()

    protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    # add "audio" and/or "duration" preprocessors when missing
    file = next(getattr(protocol, subset)())
    missing_preprocessors = dict()
    if "audio" not in file:
        missing_preprocessors["audio"] = FileFinder()
    if "duration" not in file:
        missing_preprocessors["duration"] = get_audio_duration
    if missing_preprocessors:
        preprocessors.update(missing_preprocessors)
        protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    protocol.name = protocol_name

    return protocol


def create_zip(validate_dir: Path):
    """

    # create zip file containing:
    # config.yml
    # {train_dir}/hparams.yml
    # {train_dir}/weights/epoch={epoch:04d}.ckpt
    # {validate_dir}/params.yml

    """

    existing_zips = list(validate_dir.glob("*.zip"))
    if len(existing_zips) == 1:
        existing_zips[0].unlink()
    elif len(existing_zips) > 1:
        msg = (
            f"Looks like there are too many torch.hub zip files " f"in {validate_dir}."
        )
        raise NotImplementedError(msg)

    params_yml = validate_dir / "params.yml"

    with open(params_yml, "r") as fp:
        params = yaml.load(fp, Loader=yaml.SafeLoader)
        epoch = params["epoch"]

    xp_dir = validate_dir.parents[3]
    config_yml = xp_dir / "config.yml"

    train_dir = validate_dir.parents[1]
    weights_dir = train_dir / "weights"
    hparams_yml = train_dir / "hparams.yaml"
    epoch_ckpt = weights_dir / f"epoch={epoch:04d}.ckpt"

    hub_zip = validate_dir / "hub.zip"
    with zipfile.ZipFile(hub_zip, "w") as z:
        z.write(config_yml, arcname=config_yml.relative_to(xp_dir))
        z.write(hparams_yml, arcname=hparams_yml.relative_to(xp_dir))
        z.write(params_yml, arcname=params_yml.relative_to(xp_dir))
        z.write(epoch_ckpt, arcname=epoch_ckpt.relative_to(xp_dir))

    sha256_hash = hashlib.sha256()
    with open(hub_zip, "rb") as fp:
        for byte_block in iter(lambda: fp.read(4096), b""):
            sha256_hash.update(byte_block)

    hash_prefix = sha256_hash.hexdigest()[:10]
    target = validate_dir / f"{hash_prefix}.zip"
    hub_zip.rename(target)

    return target


def get_last_epoch(train_dir: Path):
    return int(
        sorted(glob(str(train_dir) + "/weights/epoch=[0-9][0-9][0-9][0-9].ckpt"))[-1][
            -9:-5
        ]
    )


def run_train(arg: Dict):

    root_dir = Path(arg["<root>"]).expanduser().resolve(strict=True)

    pretrained = arg["--pretrained"]
    if pretrained is not None:
        pretrained_task = Wrapper(pretrained).scorer_.task_
        msg = "--pretrained option is not supported yet..."
        raise NotImplementedError(msg)
        # set and use pretrained_task.config_yml smartly

    config_yml = root_dir / "config.yml"
    # handle "--pretrained" case with no config.yml
    task_class, hparams, preprocessors = BaseTask.load_config(config_yml)

    subset = "train" if arg["--subset"] is None else arg["--subset"]
    protocol = load_protocol(
        arg["<protocol>"], subset=subset, preprocessors=preprocessors
    )

    train_dir = root_dir / "train" / f"{protocol.name}.{subset}"

    # initialize trainer
    trainer_params = dict()
    trainer_params["checkpoint_callback"] = pl.callbacks.ModelCheckpoint(
        filepath=f"{root_dir}/train/{protocol.name}.{subset}/weights/{{epoch:04d}}",
        save_top_k=-1,
        save_last=True,
        period=1,
    )
    trainer_params["logger"] = pl.loggers.TensorBoardLogger(
        f"{root_dir}/train/{protocol.name}.{subset}", name="", version=""
    )
    if getattr(hparams, "batch_size", None) == "auto":
        trainer_params["auto_scale_batch_size"] = "power"

    # TODO: set "benchmark" to True if and only if all training chunks
    # have the same size
    trainer_params["benchmark"] = False

    # TODO: try this distributed thing...
    # trainer_params["distributed_backend"] = ...
    # trainer_params["num_nodes"] = ...
    # trainer_params["num_processes"] = ...

    # TODO: add a debug option that activates this parameter
    # trainer_params["fast_dev_run"] = ...
    #trainer_params["overfit_batches"] = 1
    # trainer_params["profiler"] = ...
    # trainer_params["limit_train_batches"] = ...
    trainer_params["weights_summary"] = "full"

    # TODO: try this "precision" thing
    # trainer_params["precision"] = ...

    if arg["--gpu"]:
        trainer_params["gpus"] = -1
    elif arg["--cpu"]:
        trainer_params["gpus"] = None
    else:
        trainer_params["gpus"] = -1 if torch.cuda.is_available() else None

    if arg["--from"] is None:
        checkpoint = None
    else:
        resume_epoch = (
            get_last_epoch(train_dir) if arg["--from"] == "last" else int(arg["--from"])
        )
        checkpoint = str(train_dir / "weights" / f"epoch={resume_epoch:04d}.ckpt")
    trainer_params["resume_from_checkpoint"] = checkpoint

    trainer_params["max_epochs"] = int(arg["--to"]) + 1

    trainer_params["distributed_backend"] = (
        "ddp" if torch.cuda.device_count() > 1 else None
    )

    trainer = pl.Trainer(**trainer_params)

    num_workers = (
        max(1, multiprocessing.cpu_count() // 2)
        if arg["--parallel"] is None
        else int(arg["--parallel"])
    )

    if getattr(hparams, "learning_rate", "auto") == "auto":

        # initialize model with an arbitray learning rate (or lr_find will complain)
        hparams.learning_rate = 1e-3
        task = task_class(
            hparams,
            protocol=protocol,
            subset=subset,
            train_dir=train_dir,
            num_workers=num_workers,
        )

        # suggest good learning rate
        lr_finder = trainer.lr_find(task, min_lr=1e-7, max_lr=10, num_training=1000)
        suggested_lr = lr_finder.suggestion()

        # initialize model with suggested learning rate
        hparams.learning_rate = suggested_lr
        task = task_class(hparams, train_dir=train_dir, num_workers=num_workers)

    else:
        task = task_class(
            hparams,
            protocol=protocol,
            subset=subset,
            train_dir=train_dir,
            num_workers=num_workers,
        )

    trainer.fit(task)


def run_validate(arg):
    train_dir = Path(arg["<train>"]).expanduser().resolve(strict=True)
    hparams_yml = train_dir / "hparams.yaml"

    root_dir = train_dir.parents[1]
    config_yml = root_dir / "config.yml"

    start = arg["--from"]
    if start is None:
        start = 0
    elif start == "last":
        start = get_last_epoch(train_dir)
    else:
        start = int(start)

    end = arg["--to"]
    if end == "last":
        end = get_last_epoch(train_dir) + 1
    else:
        end = int(end) + 1

    every = int(arg["--every"])

    task_class, hparams, preprocessors = BaseTask.load_config(
        config_yml, hparams_yml=hparams_yml
    )

    duration = None if arg["--duration"] is None else float(arg["--duration"])
    step = None if arg["--step"] is None else float(arg["--step"])
    batch_size = None if arg["--batch"] is None else int(arg["--batch"])
    if arg["--gpu"]:
        device = "cuda"
    elif arg["--cpu"]:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    protocol_name = arg["<protocol>"]
    subset = "development" if arg["--subset"] is None else arg["--subset"]
    protocol = load_protocol(
        arg["<protocol>"], subset=subset, preprocessors=preprocessors
    )
    files = list(getattr(protocol, subset)())

    criterion = task_class.validation_criterion(protocol)
    validate_dir = train_dir / f"validate_{criterion}" / f"{protocol_name}.{subset}"
    validate_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(validate_dir), purge_step=start)

    params_yml = validate_dir / "params.yml"
    if params_yml.exists():
        with open(params_yml, "r") as fp:
            params = yaml.load(fp, Loader=yaml.SafeLoader)
        best_epoch = params["epoch"]
        best_value = params[criterion]
        best_params = params.get("params", None)
    else:
        best_epoch = None
        best_value = None
        best_params = None

    past_value = None
    past_epoch = None

    # run feature extraction once and for all
    task = task_class(hparams)
    if not isinstance(task.feature_extraction, (Precomputed, RawAudio)):
        tqdm_files = tqdm(
            files, desc="Feature extraction", unit="file", position=0, leave=False
        )
        for file in tqdm_files:
            file["features"] = task.feature_extraction(file)

    pbar = trange(start, end, every, unit="epoch", position=0, leave=True)
    for e, epoch in enumerate(pbar):

        if best_epoch is None:
            best_prefix = ""
        else:
            best_prefix = (
                f"{criterion} = {100 * best_value:g}% @ epoch #{best_epoch} | "
            )

        if e > 0:
            past_prefix = f"{100 * past_value:g}% @ epoch #{past_epoch}"
        else:
            past_prefix = ""

        # wait until epoch currently validated is available
        checkpoint_path = train_dir / "weights" / f"epoch={epoch:04d}.ckpt"
        if not checkpoint_path.exists():
            description = (
                best_prefix
                + past_prefix
                + f" | waiting for training to reach epoch #{epoch}..."
            )
            pbar.set_description(description)
            while not checkpoint_path.exists():
                time.sleep(10)

        description = best_prefix + past_prefix
        pbar.set_description(description)

        # apply pretrained model
        pretrained = Pretrained(
            validate_dir,
            epoch=epoch,
            duration=duration,
            step=step,
            device=device,
            batch_size=batch_size,
        )
        tqdm_files = tqdm(
            files,
            desc=f"epoch #{epoch} | extracting raw scores...",
            unit="file",
            position=1,
            leave=False,
        )
        for file in tqdm_files:
            file["scores"] = pretrained(file)

        details = task.validation(
            files, warm_start=best_params, epoch=epoch, protocol=protocol, subset=subset
        )

        value = details["value"]
        direction = 1 if details["minimize"] else -1
        if best_value is None or (direction * value < direction * best_value):
            best_value = value
            best_epoch = epoch

            new_best = {
                details["metric"]: value,
                "epoch": best_epoch,
            }

            if "params" in details:
                new_best["params"] = details["params"]

            with open(params_yml, "w") as fp:
                fp.write(yaml.dump(new_best, default_flow_style=False))

            # create/update zip file for later upload to torch.hub
            _ = create_zip(validate_dir)

        writer.add_scalar(
            f"validate/{criterion}/{protocol_name}.{subset}", value, global_step=epoch,
        )

        past_value = value
        past_epoch = epoch


def run_apply(arg):

    validate_dir = Path(arg["<validate>"]).expanduser().resolve(strict=True)

    duration = None if arg["--duration"] is None else float(arg["--duration"])
    step = None if arg["--step"] is None else float(arg["--step"])
    batch_size = None if arg["--batch"] is None else int(arg["--batch"])
    if arg["--gpu"]:
        device = "cuda"
    elif arg["--cpu"]:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_pretrained = arg["--pretrained"]

    if use_pretrained is None:
        pretrained = Pretrained(
            validate_dir=validate_dir,
            duration=duration,
            step=step,
            batch_size=batch_size,
            device=device,
        )
        output_dir = validate_dir / "apply" / f"{pretrained.epoch_:04d}"

    else:

        if use_pretrained in torch.hub.list("pyannote/pyannote-audio"):
            output_dir = validate_dir / use_pretrained
        else:
            output_dir = validate_dir

        pretrained = Wrapper(
            use_pretrained,
            duration=duration,
            step=step,
            batch_size=batch_size,
            device=device,
        )

    precomputed_params = {}
    if hasattr(pretrained, "classes"):
        precomputed_params["classes"] = pretrained.classes
    if hasattr(pretrained, "dimension"):
        precomputed_params["dimension"] = pretrained.dimension
    precomputed = Precomputed(
        root_dir=output_dir,
        sliding_window=pretrained.sliding_window,
        **precomputed_params,
    )

    subset = "test" if arg["--subset"] is None else arg["--subset"]
    protocol = load_protocol(arg["<protocol>"], subset=subset,)
    files = list(getattr(protocol, subset)())

    for file in tqdm(iterable=files, desc=f"{subset.title()}", unit="file"):
        file["scores"] = pretrained(file)
        precomputed.dump(file, file["scores"])

    try:
        pipeline = pretrained.task_.validation_pipeline()
    except AttributeError:
        return
    pipeline.instantiate(pretrained.pipeline_params_)

    # load pipeline metric (when available)
    try:
        metric = pipeline.get_metric()
    except NotImplementedError:
        metric = None

    # apply pipeline and dump output to RTTM files
    output_rttm = output_dir / f"{protocol.name}.{subset}.rttm"
    with open(output_rttm, "w") as fp:
        for file in tqdm(iterable=files, desc=f"{subset.title()}", unit="file"):
            hypothesis = pipeline(file)
            pipeline.write_rttm(fp, hypothesis)

            # compute evaluation metric (when possible)
            if "annotation" not in file:
                metric = None

            # compute evaluation metric (when available)
            if metric is None:
                continue

            reference = file["annotation"]
            uem = get_annotated(file)
            _ = metric(reference, hypothesis, uem=uem)

    # print pipeline metric (when available)
    if metric is None:
        return

    output_eval = output_dir / f"{protocol.name}.{subset}.eval"
    with open(output_eval, "w") as fp:
        fp.write(str(metric))


def main():

    arg = docopt(__doc__, version=f"pyannote-audio {__version__}")

    if arg["--debug"]:
        msg = "Debug mode is enabled, this option might slow execution considerably."
        warnings.warn(msg, RuntimeWarning)
        torch.autograd.set_detect_anomaly(True)

    if arg["train"]:
        run_train(arg)

    elif arg["validate"]:
        run_validate(arg)

    elif arg["apply"]:
        run_apply(arg)


if __name__ == "__main__":
    main()
