# The MIT License (MIT)

# Copyright (c) 2018- CNRS

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
Pipeline

Usage:
  pyannote-pipeline train [options] [(--forever | --iterations=<iterations>)] <experiment_dir> <database.task.protocol>
  pyannote-pipeline best [options] <experiment_dir> <database.task.protocol>
  pyannote-pipeline apply [options] <train_dir> <database.task.protocol>
  pyannote-pipeline -h | --help
  pyannote-pipeline --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
  --registry=<db.yml>        Path to, comma-separated, database configuration files.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset. Defaults to 'development' in "train"
                             mode, and to 'test' in "apply" mode.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  --iterations=<iterations>  Number of iterations. [default: 1]
  --forever                  Iterate forever.
  --sampler=<sampler>        Choose sampler between RandomSampler or TPESampler
                             Defaults to Optuna's recommended sampler.
  --pruner=<pruner>          Choose pruner between MedianPruner or
                             SuccessiveHalvingPruner. Defaults to no pruning.
  --pretrained=<train_dir>   Use parameters in existing training directory to
                             bootstrap the optimization process. In practice,
                             this will simply run a first trial with this set
                             of parameters.
  --average-case             Optimize for average case instead of worst case.

"apply" mode:
  <train_dir>                Path to the directory containing trained hyper-
                             parameters (i.e. the output of "train" mode).

  --use-filter               Apply pipeline only to files that pass the filter.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml that describes the pipeline.

    ................... <experiment_dir>/config.yml ...................
    pipeline:
       name: Yin2018
       params:
          sad: tutorials/pipeline/sad
          scd: tutorials/pipeline/scd
          emb: tutorials/pipeline/emb
          metric: angular

    # preprocessors can be used to automatically add keys into
    # each (dict) file obtained from pyannote.database protocols.
    preprocessors:
       audio: ~/.pyannote/db.yml   # load template from YAML file
       video: ~/videos/{uri}.mp4   # define template directly

    # filters can be used to filter out some files from the protocol
    # (e.g. to only keep files with a specific number of speakers)
    filters:
        pyannote.audio.utils.protocol.FilterByNumberOfSpeakers:
            num_speakers: 2

    # one can freeze some hyper-parameters if needed (e.g. when
    # only part of the pipeline needs to be updated)
    freeze:
       speech_turn_segmentation:
          speech_activity_detection:
              onset: 0.5
              offset: 0.5

    # pyannote.audio pipelines will run on CPU by default.
    # use `device` key to send it to GPU.
    device: cuda
    ...................................................................

"train" mode:
    Tune the pipeline hyper-parameters
        <experiment_dir>/<database.task.protocol>.<subset>.yml

"best" mode:
    Display current best loss and corresponding hyper-paramters.

"apply" mode
    Apply the pipeline (with best set of hyper-parameters)

"""

import itertools
import os
import os.path
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from docopt import docopt
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, get_annotated, registry
from tqdm import tqdm

from .optimizer import Optimizer


class Experiment:
    """Pipeline experiment

    Parameters
    ----------
    experiment_dir : `Path`
        Experiment root directory.
    training : `bool`, optional
        Switch to training mode
    """

    CONFIG_YML = "{experiment_dir}/config.yml"
    TRAIN_DIR = "{experiment_dir}/train/{protocol}.{subset}"
    APPLY_DIR = "{train_dir}/apply/{date}"

    @classmethod
    def from_train_dir(cls, train_dir: Path, training: bool = False) -> "Experiment":
        """Load pipeline from train directory

        Parameters
        ----------
        train_dir : `Path`
            Path to train directory
        training : `bool`, optional
            Switch to training mode.

        Returns
        -------
        xp : `Experiment`
            Pipeline experiment.
        """
        experiment_dir = train_dir.parents[1]
        xp = cls(experiment_dir, training=training)
        params_yml = train_dir / "params.yml"
        xp.mtime_ = datetime.fromtimestamp(os.path.getmtime(params_yml))
        xp.pipeline_.load_params(params_yml)
        return xp

    def __init__(self, experiment_dir: Path, training: bool = False):
        super().__init__()

        self.experiment_dir = experiment_dir

        # load configuration file
        config_yml = self.CONFIG_YML.format(experiment_dir=self.experiment_dir)
        with open(config_yml, "r") as fp:
            self.config_ = yaml.load(fp, Loader=yaml.SafeLoader)

        # initialize preprocessors
        preprocessors = {}
        for key, preprocessor in self.config_.get("preprocessors", {}).items():
            # preprocessors:
            #    key:
            #       name: package.module.ClassName
            #       params:
            #          param1: value1
            #          param2: value2
            if isinstance(preprocessor, dict):
                Klass = get_class_by_name(
                    preprocessor["name"], default_module_name="pyannote.pipeline"
                )
                preprocessors[key] = Klass(**preprocessor.get("params", {}))
                continue

            try:
                # preprocessors:
                #    key: /path/to/database.yml
                preprocessors[key] = FileFinder(database_yml=preprocessor)

            except FileNotFoundError:
                # preprocessors:
                #    key: /path/to/{uri}.wav
                template = preprocessor
                preprocessors[key] = template

        self.preprocessors_ = preprocessors

        # initialize filters
        filters = []
        for key, params in self.config_.get("filters", {}).items():
            Klass = get_class_by_name(key)
            filters.append(Klass(**params))

        def all_filters(i) -> bool:
            return all(f(i) for f in filters)

        self.filters_ = all_filters

        # initialize pipeline
        pipeline_name = self.config_["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        self.pipeline_ = Klass(**self.config_["pipeline"].get("params", {}))

        # freeze parameters
        if "freeze" in self.config_:
            params = self.config_["freeze"]
            self.pipeline_.freeze(params)

        # send to device
        if "device" in self.config_:
            import torch

            device = torch.device(self.config_["device"])
            self.pipeline_.to(device)

    def train(
        self,
        protocol_name: str,
        subset: str | None = "development",
        pretrained: Path | None = None,
        n_iterations: int = 1,
        sampler: str | None = None,
        pruner: str | None = None,
        average_case: bool = False,
    ):
        """Train pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol to use.
        subset : `str`, optional
            Use this subset for training. Defaults to 'development'.
        pretrained : Path, optional
            Use parameters in "pretrained" training directory to bootstrap the
            optimization process. In practice this will simply run a first trial
            with this set of parameters.
        n_iterations : `int`, optional
            Number of iterations. Defaults to 1.
        sampler : `str`, optional
            Choose sampler between RandomSampler and TPESampler. Defaults to
            Optuna's recommended sampler for the objective mode.
        pruner : `str`, optional
            Choose between MedianPruner or SuccessiveHalvingPruner.
        average_case : `bool`, optional
            Optimise for average case. Defaults to False (i.e. worst case).
        """
        train_dir = Path(
            self.TRAIN_DIR.format(
                experiment_dir=self.experiment_dir,
                protocol=protocol_name,
                subset=subset,
            )
        )
        train_dir.mkdir(parents=True, exist_ok=True)

        protocol = registry.get_protocol(
            protocol_name, preprocessors=self.preprocessors_
        )

        study_name = "default"
        optimizer = Optimizer(
            self.pipeline_,
            db=train_dir / "trials.journal",
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            average_case=average_case,
        )

        direction = 1 if self.pipeline_.get_direction() == "minimize" else -1

        params_yml = train_dir / "params.yml"

        progress_bar = tqdm(unit="trial", position=0, leave=True)
        progress_bar.set_description("First trial in progress")
        progress_bar.update(0)

        if pretrained:
            pre_params_yml = pretrained / "params.yml"
            with open(pre_params_yml, mode="r") as fp:
                pre_params = yaml.load(fp, Loader=yaml.SafeLoader)
            warm_start = pre_params["params"]

        else:
            warm_start = None

        inputs = list(filter(self.filters_, getattr(protocol, subset)()))

        iterations = optimizer.tune_iter(
            inputs, warm_start=warm_start, show_progress=True
        )

        try:
            best_loss = optimizer.best_loss
        except ValueError:
            best_loss = direction * np.inf
        count = itertools.count() if n_iterations < 0 else range(n_iterations)

        for i, status in zip(count, iterations):
            loss = status["loss"]

            if direction * loss < direction * best_loss:
                best_params = status["params"]
                best_loss = loss
                self.pipeline_.dump_params(
                    params_yml, params=best_params, loss=best_loss
                )

            # progress bar
            desc = f"Best trial: {100 * best_loss:g}%"
            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

    def best(self, protocol_name: str, subset: str = "development"):
        """Print current best pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol used for training.
        subset : `str`, optional
            Subset used for training. Defaults to 'development'.
        """

        train_dir = Path(
            self.TRAIN_DIR.format(
                experiment_dir=self.experiment_dir,
                protocol=protocol_name,
                subset=subset,
            )
        )

        study_name = "default"
        optimizer = Optimizer(
            self.pipeline_, db=train_dir / "trials.journal", study_name=study_name
        )

        try:
            best_loss = optimizer.best_loss
        except ValueError:
            print("Still waiting for at least one iteration to succeed.")
            return

        best_params = optimizer.best_params

        print(f"Loss = {100 * best_loss:g}% with the following hyper-parameters:")

        content = yaml.dump(best_params, default_flow_style=False)
        print(content)

    def apply(
        self,
        protocol_name: str,
        output_dir: Path,
        subset: str | None = "test",
        use_filter: bool = False,
    ):
        """Apply current best pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol to process.
        subset : `str`, optional
            Subset to process. Defaults to 'test'
        """

        # file generator
        protocol = registry.get_protocol(
            protocol_name, preprocessors=self.preprocessors_
        )

        # load pipeline metric (when available)
        try:
            metric = self.pipeline_.get_metric()
        except NotImplementedError:
            metric = None

        output_dir.mkdir(parents=True, exist_ok=True)
        if use_filter:
            output_ext = (
                output_dir
                / f"{protocol_name}.{subset}_INCOMPLETE.{self.pipeline_.write_format}"
            )
        else:
            output_ext = (
                output_dir / f"{protocol_name}.{subset}.{self.pipeline_.write_format}"
            )

        with open(output_ext, mode="w") as fp:
            files = list(getattr(protocol, subset)())
            if use_filter:
                files = filter(self.filters_, files)

            desc = f"Processing {protocol_name} ({subset})"
            for current_file in tqdm(iterable=files, desc=desc, unit="file"):
                # apply pipeline and dump output to file
                output = self.pipeline_(current_file)
                self.pipeline_.write(fp, output)

                # compute evaluation metric (when possible)
                reference = current_file.get("annotation", None)
                if reference is None:
                    metric = None

                # compute evaluation metric (when available)
                if metric is None:
                    continue

                uem = get_annotated(current_file)
                _ = metric(reference, output, uem=uem)

        # "latest" symbolic link
        latest = output_dir.parent / "latest"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(output_dir)

        # print pipeline metric (when available)
        if metric is None:
            msg = (
                f"For some (possibly good) reason, the output of this "
                f"pipeline could not be evaluated on {protocol_name}."
            )
            print(msg)
            return

        if use_filter:
            output_eval = output_dir / f"{protocol_name}.{subset}_INCOMPLETE.eval"
        else:
            output_eval = output_dir / f"{protocol_name}.{subset}.eval"

        with open(output_eval, "w") as fp:
            fp.write(str(metric))


def main():
    arguments = docopt(__doc__, version="Tunable pipelines")

    for database_yml in arguments["--registry"].split(","):
        registry.load_database(database_yml)

    protocol_name = arguments["<database.task.protocol>"]
    subset = arguments["--subset"]

    if arguments["train"]:
        if subset is None:
            subset = "development"

        if arguments["--forever"]:
            iterations = -1
        else:
            iterations = int(arguments["--iterations"])

        sampler = arguments["--sampler"]
        pruner = arguments["--pruner"]

        pretrained = arguments["--pretrained"]
        if pretrained:
            pretrained = Path(pretrained).expanduser().resolve(strict=True)

        average_case = arguments["--average-case"]

        experiment_dir = Path(arguments["<experiment_dir>"])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        experiment = Experiment(experiment_dir, training=True)
        experiment.train(
            protocol_name,
            subset=subset,
            n_iterations=iterations,
            pretrained=pretrained,
            sampler=sampler,
            pruner=pruner,
            average_case=average_case,
        )

    if arguments["best"]:
        if subset is None:
            subset = "development"

        experiment_dir = Path(arguments["<experiment_dir>"])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        experiment = Experiment(experiment_dir, training=False)
        experiment.best(protocol_name, subset=subset)

    if arguments["apply"]:
        if subset is None:
            subset = "test"

        use_filter = arguments["--use-filter"]

        train_dir = Path(arguments["<train_dir>"])
        train_dir = train_dir.expanduser().resolve(strict=True)
        experiment = Experiment.from_train_dir(train_dir, training=False)

        output_dir = Path(
            experiment.APPLY_DIR.format(
                train_dir=train_dir, date=experiment.mtime_.strftime("%Y%m%d-%H%M%S")
            )
        )

        experiment.apply(
            protocol_name, output_dir, subset=subset, use_filter=use_filter
        )
