# The MIT License (MIT)

# Copyright (c) 2018-2021 CNRS

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
# Hadrien TITEUX

import time
import warnings
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import optuna.logging
import optuna.pruners
import optuna.samplers
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial, Trial
from scipy.stats import bayes_mvs
from tqdm import tqdm

from .pipeline import Pipeline
from .typing import Direction, Objective, PipelineInput

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Optimizer:
    """Pipeline optimizer

    Parameters
    ----------
    pipeline : `Pipeline`
        Pipeline.
    db : `Path`, optional
        Path to trial database on disk. Use ".sqlite" extension for SQLite
        backend, and ".journal" for Journal backend (prefered for parallel
        optimization).
    study_name : `str`, optional
        Name of study. In case it already exists, study will continue from
        there. # TODO -- generate this automatically
    sampler : `str` or sampler instance, optional
        Algorithm for value suggestion. Must be one of "RandomSampler" or
        "TPESampler", or a sampler instance. When omitted, Optuna chooses its
        recommended sampler for the study's objective mode ("TPESampler" for
        both single- and multi-objective studies in Optuna 5).
    pruner : `str` or pruner instance, optional
        Algorithm for early pruning of trials. Must be one of "MedianPruner" or
        "SuccessiveHalvingPruner", or a pruner instance.
        Defaults to no pruning.
    seed : `int`, optional
        Seed value for the random number generator of the sampler.
        Defaults to no seed.
    average_case : `bool`, optional
        Optimize for average case (default).
        Set to False to optimize for worst case.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        db: Path | None = None,
        study_name: str | None = None,
        sampler: str | BaseSampler | None = None,
        pruner: str | BasePruner | None = None,
        seed: int | None = None,
        average_case: bool = True,
    ):
        self.pipeline = pipeline

        self.db = db
        if db is None:
            self.storage_ = None
        else:
            extension = Path(self.db).suffix
            if extension == ".db":
                warnings.warn(
                    "Storage with '.db' extension has been deprecated. Use '.sqlite' instead."
                )
                self.storage_ = RDBStorage(f"sqlite:///{self.db}")
            elif extension == ".sqlite":
                self.storage_ = RDBStorage(f"sqlite:///{self.db}")
            elif extension == ".journal":
                self.storage_ = JournalStorage(JournalFileBackend(f"{self.db}"))
        self.study_name = study_name

        if isinstance(sampler, BaseSampler):
            self.sampler = sampler
        elif isinstance(sampler, str):
            try:
                self.sampler = getattr(optuna.samplers, sampler)(seed=seed)
            except AttributeError:
                msg = '`sampler` must be one of "RandomSampler" or "TPESampler"'
                raise ValueError(msg)
        elif sampler is None:
            if seed is not None:
                self.sampler = TPESampler(seed=seed)
            else:
                # Delegate sampler selection to Optuna so that single- and
                # multi-objective studies use its recommended defaults.
                self.sampler = None

        if isinstance(pruner, BasePruner):
            self.pruner = pruner
        elif isinstance(pruner, str):
            try:
                self.pruner = getattr(optuna.pruners, pruner)()
            except AttributeError:
                msg = '`pruner` must be one of "MedianPruner" or "SuccessiveHalvingPruner"'
                raise ValueError(msg)
        else:
            self.pruner = None

        directions = self.pipeline.get_direction()
        if isinstance(directions, str):
            directions = (directions,)
        else:
            directions = tuple(directions)

        if not directions:
            raise ValueError("`pipeline.get_direction()` must not be empty.")

        if any(direction not in {"minimize", "maximize"} for direction in directions):
            raise ValueError(
                "`pipeline.get_direction()` must return 'minimize', 'maximize', "
                "or a non-empty sequence containing those values."
            )

        self.directions: tuple[Direction, ...] = directions
        self.multi_objective = len(self.directions) > 1

        if self.multi_objective and self.pruner is not None:
            raise ValueError(
                "Optuna does not support trial pruning for multi-objective studies."
            )

        # generate name of study based on pipeline hash
        # Klass = pipeline.__class__
        # study_name = f'{Klass.__module__}.{Klass.__name__}[{hash(pipeline)}]'

        study_kwargs = {
            "study_name": self.study_name,
            "load_if_exists": True,
            "storage": self.storage_,
            "sampler": self.sampler,
            "pruner": self.pruner,
        }
        if self.multi_objective:
            study_kwargs["directions"] = self.directions
        else:
            study_kwargs["direction"] = self.directions[0]

        self.study_ = optuna.create_study(**study_kwargs)
        self.sampler = self.study_.sampler

        if self.multi_objective:
            try:
                metrics = self._as_sequence(self.pipeline.get_metric())
            except NotImplementedError:
                metrics = None

            if metrics is not None:
                self._validate_objective_count(metrics, source="metrics")
                metric_names = [
                    getattr(metric, "name", metric.__class__.__name__)
                    for metric in metrics
                ]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ExperimentalWarning)
                    self.study_.set_metric_names(metric_names)

        self.average_case = average_case

    @property
    def best_loss(self) -> float:
        """Return best loss so far"""
        if self.multi_objective:
            raise RuntimeError(
                "Multi-objective studies have no single best loss. "
                "Use `pareto_front` instead."
            )

        try:
            best_value = self.study_.best_value
        except Exception:
            direction: int = 1 if self.directions[0] == "minimize" else -1
            best_value = direction * np.inf
        return best_value

    @property
    def best_params(self) -> dict:
        """Return best parameters so far"""
        if self.multi_objective:
            raise RuntimeError(
                "Multi-objective studies have no single best set of parameters. "
                "Use `pareto_front` instead."
            )

        trial = FixedTrial(self.study_.best_params)
        return self.pipeline.parameters(trial=trial)

    @property
    def best_pipeline(self) -> Pipeline:
        """Return pipeline instantiated with best parameters so far"""
        if self.multi_objective:
            raise RuntimeError(
                "Multi-objective studies have no single best pipeline. "
                "Choose one entry from `pareto_front` and instantiate its parameters."
            )

        return self.pipeline.instantiate(self.best_params)

    @staticmethod
    def _as_sequence(value) -> tuple:
        """Normalize scalar or sequence objective values to a tuple."""
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(value)
        return (value,)

    def _validate_objective_count(self, values: Sequence, source: str = "values"):
        """Check that values and optimization directions have matching sizes."""
        if len(values) != len(self.directions):
            raise ValueError(
                f"Pipeline returned {len(values)} {source}, but "
                f"`pipeline.get_direction()` returned {len(self.directions)} "
                "directions."
            )

    @property
    def pareto_front(self) -> list[dict]:
        """Return Pareto-optimal trials with nested pipeline parameters.

        Returns
        -------
        pareto_front : list of dict
            Each entry contains the Optuna trial number, its objective values,
            and the corresponding nested pipeline parameters. Returns an empty
            list until at least one trial completes.
        """
        if not self.multi_objective:
            raise RuntimeError(
                "`pareto_front` is only available for multi-objective studies. "
                "Use `best_loss` and `best_params` instead."
            )

        return [
            {
                "number": trial.number,
                "values": tuple(trial.values),
                "params": self.pipeline.parameters(trial=FixedTrial(trial.params)),
            }
            for trial in self.study_.best_trials
        ]

    def get_objective(
        self,
        inputs: Iterable[PipelineInput],
        show_progress: bool | dict = False,
    ) -> Callable[[Trial], Objective]:
        """
        Create objective function used by optuna

        Parameters
        ----------
        inputs : `iterable`
            List of inputs to process.
        show_progress : bool or dict
            Show within-trial progress bar using tqdm progress bar.
            Can also be a **kwarg dict passed to tqdm.

        Returns
        -------
        objective : `callable`
            Callable that takes trial as input and returns correspond loss.
        """

        # this is needed for `inputs` that can be only iterated once.
        inputs = list(inputs)
        len(inputs)

        if show_progress == True:
            show_progress = {"desc": "Current trial", "leave": False, "position": 1}

        def objective(trial: Trial) -> Objective:
            """Compute objective value or values

            Parameter
            ---------
            trial : `Trial`
                Current trial

            Returns
            -------
            loss : `float` or tuple
                One value for single-objective optimization, or one value per
                objective for multi-objective optimization.
            """

            # use pyannote.metrics metric when available
            try:
                metrics = self._as_sequence(self.pipeline.get_metric())
                self._validate_objective_count(metrics, source="metrics")
            except NotImplementedError:
                metrics = None
                losses = [[] for _ in self.directions]

            processing_time = []
            evaluation_time = []

            # instantiate pipeline with value suggested in current trial
            pipeline = self.pipeline.instantiate(self.pipeline.parameters(trial=trial))

            if show_progress != False:
                progress_bar = tqdm(total=len(inputs), **show_progress)
                progress_bar.update(0)

            # accumulate loss for each input
            for i, input in enumerate(inputs):
                # process input with pipeline
                # (and keep track of processing time)
                before_processing = time.time()

                # get optional kwargs to be passed to the pipeline
                # (e.g. num_speakers for speaker diarization). they
                # must be stored in a 'pipeline_kwargs' key in the
                # `input` dictionary.
                if isinstance(input, Mapping):
                    pipeline_kwargs = input.get("pipeline_kwargs", {})
                else:
                    pipeline_kwargs = {}
                output = pipeline(input, **pipeline_kwargs)
                after_processing = time.time()
                processing_time.append(after_processing - before_processing)

                # evaluate output (and keep track of evaluation time)
                before_evaluation = time.time()

                # when metric is not available, use loss method instead
                if metrics is None:
                    current_losses = self._as_sequence(pipeline.loss(input, output))
                    self._validate_objective_count(current_losses, source="loss values")
                    for objective_losses, loss in zip(losses, current_losses):
                        objective_losses.append(loss)

                # when metric is available,`input` is expected to be provided
                # by a `pyannote.database` protocol
                else:
                    from pyannote.database import get_annotated

                    for metric in metrics:
                        _ = metric(
                            input["annotation"], output, uem=get_annotated(input)
                        )

                after_evaluation = time.time()
                evaluation_time.append(after_evaluation - before_evaluation)

                if show_progress != False:
                    progress_bar.update(1)

                if self.pruner is None:
                    continue

                trial.report(
                    np.mean(losses[0]) if metrics is None else abs(metrics[0]), i
                )
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if show_progress != False:
                progress_bar.close()

            trial.set_user_attr("processing_time", sum(processing_time))
            trial.set_user_attr("evaluation_time", sum(evaluation_time))

            estimates = []
            if metrics is None:
                for objective_losses in losses:
                    if len(np.unique(objective_losses)) == 1:
                        mean = lower_bound = upper_bound = objective_losses[0]
                    else:
                        (mean, (lower_bound, upper_bound)), _, _ = bayes_mvs(
                            objective_losses, alpha=0.9
                        )
                    estimates.append((mean, lower_bound, upper_bound))
            else:
                estimates = [
                    (abs(metric), *metric.confidence_interval(alpha=0.9)[1])
                    for metric in metrics
                ]

            if self.average_case:
                values = tuple(mean for mean, _, _ in estimates)
            else:
                values = tuple(
                    upper_bound if direction == "minimize" else lower_bound
                    for direction, (_, lower_bound, upper_bound) in zip(
                        self.directions, estimates
                    )
                )

            return values if self.multi_objective else values[0]

        return objective

    def tune(
        self,
        inputs: Iterable[PipelineInput],
        n_iterations: int = 10,
        warm_start: dict | None = None,
        show_progress: bool | dict = True,
    ) -> dict:
        """Tune pipeline

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.
        n_iterations : int, optional
            Number of iterations. Defaults to 10.
        warm_start : dict, optional
            Nested dictionary of initial parameters used to bootstrap tuning.

        Returns
        -------
        result : dict
            Single-objective results contain ``loss`` and ``params`` (a nested
            dictionary of optimal parameters). Multi-objective results contain
            ``pareto_front`` instead.
        """

        # pipeline is currently being optimized
        self.pipeline.training = True

        objective = self.get_objective(inputs, show_progress=show_progress)

        if warm_start:
            flattened_params = self.pipeline._flatten(warm_start)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ExperimentalWarning)
                self.study_.enqueue_trial(flattened_params)

        self.study_.optimize(objective, n_trials=n_iterations, timeout=None, n_jobs=1)

        # pipeline is no longer being optimized
        self.pipeline.training = False

        if self.multi_objective:
            return {"pareto_front": self.pareto_front}

        return {"loss": self.best_loss, "params": self.best_params}

    def tune_iter(
        self,
        inputs: Iterable[PipelineInput],
        warm_start: dict | None = None,
        show_progress: bool | dict = True,
    ) -> Generator[dict, None, None]:
        """

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.
        warm_start : dict, optional
            Nested dictionary of initial parameters used to bootstrap tuning.

        Yields
        ------
        result : dict
            Single-objective results contain ``loss`` and ``params`` (a nested
            dictionary of optimal parameters). Multi-objective results contain
            ``pareto_front`` instead.
        """

        objective = self.get_objective(inputs, show_progress=show_progress)

        if warm_start:
            flattened_params = self.pipeline._flatten(warm_start)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ExperimentalWarning)
                self.study_.enqueue_trial(flattened_params)

        while True:
            # pipeline is currently being optimized
            self.pipeline.training = True

            # one trial at a time
            self.study_.optimize(objective, n_trials=1, timeout=None, n_jobs=1)

            if self.multi_objective:
                pareto_front = self.pareto_front
                if not pareto_front:
                    continue

                self.pipeline.training = False
                yield {"pareto_front": pareto_front}
                continue

            try:
                best_loss = self.best_loss
                best_params = self.best_params
            except ValueError:
                continue

            # pipeline is no longer being optimized
            self.pipeline.training = False

            yield {"loss": best_loss, "params": best_params}
