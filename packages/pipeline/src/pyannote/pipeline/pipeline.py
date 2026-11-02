# The MIT License (MIT)

# Copyright (c) 2018-2022 CNRS

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

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO, Union

import yaml
from filelock import FileLock
from optuna.trial import Trial
from pyannote.core import Annotation, Timeline

from .typing import Directions, PipelineInput, PipelineOutput


class Pipeline:
    """Base tunable pipeline"""

    def __init__(self):

        # un-instantiated parameters (= `Parameter` instances)
        self._parameters: dict[str, Parameter] = OrderedDict()

        # instantiated parameters
        self._instantiated: dict[str, Any] = OrderedDict()

        # sub-pipelines
        self._pipelines: dict[str, Pipeline] = OrderedDict()

        # whether pipeline is currently being optimized
        self.training = False

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, training):
        self._training = training
        # recursively set sub-pipeline training attribute
        for pipeline in self._pipelines.values():
            pipeline.training = training

    def __hash__(self):
        # FIXME -- also keep track of (sub)pipeline attributes
        frozen = self.parameters(frozen=True)
        return hash(tuple(sorted(self._flatten(frozen).items())))

    def __getattr__(self, name):
        """(Advanced) attribute getter"""

        # in case `name` corresponds to an instantiated parameter value, returns it
        if "_instantiated" in self.__dict__:
            _instantiated = self.__dict__["_instantiated"]
            if name in _instantiated:
                return _instantiated[name]

        # in case `name` corresponds to a parameter, returns it
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]

        # in case `name` corresponds to a sub-pipeline, returns it
        if "_pipelines" in self.__dict__:
            _pipelines = self.__dict__["_pipelines"]
            if name in _pipelines:
                return _pipelines[name]

        msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def __setattr__(self, name, value):
        """(Advanced) attribute setter

        If `value` is an instance of `Parameter`, store it in `_parameters`.
        elif `value` is an instance of `Pipeline`, store it in `_pipelines`.
        elif `value` isn't an instance of `Parameter` and `name` is in `_parameters`,
        store `value` in `_instantiated`.
        """

        # imported here to avoid circular import
        from .parameter import Parameter

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        _parameters = self.__dict__.get("_parameters")
        _instantiated = self.__dict__.get("_instantiated")
        _pipelines = self.__dict__.get("_pipelines")

        # if `value` is an instance of `Parameter`, store it in `_parameters`

        if isinstance(value, Parameter):
            if _parameters is None:
                msg = "cannot assign hyper-parameters before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(self.__dict__, _instantiated, _pipelines)
            _parameters[name] = value
            return

        # add/update one sub-pipeline
        if isinstance(value, Pipeline):
            if _pipelines is None:
                msg = "cannot assign sub-pipelines before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(self.__dict__, _parameters, _instantiated)
            _pipelines[name] = value
            return

        # store instantiated parameter value
        if _parameters is not None and name in _parameters:
            _instantiated[name] = value
            return

        object.__setattr__(self, name, value)

    def __delattr__(self, name):

        if name in self._parameters:
            del self._parameters[name]

        elif name in self._instantiated:
            del self._instantiated[name]

        elif name in self._pipelines:
            del self._pipelines[name]

        else:
            object.__delattr__(self, name)

    def _flattened_parameters(
        self, frozen: bool | None = False, instantiated: bool | None = False
    ) -> dict:
        """Get flattened dictionary of parameters

        Parameters
        ----------
        frozen : `bool`, optional
            Only return value of frozen parameters.
        instantiated : `bool`, optional
            Only return value of instantiated parameters.

        Returns
        -------
        params : `dict`
            Flattened dictionary of parameters.
        """

        # imported here to avoid circular imports
        from .parameter import Frozen

        if frozen and instantiated:
            msg = "one must choose between `frozen` and `instantiated`."
            raise ValueError(msg)

        # initialize dictionary with root parameters
        if instantiated:
            params = dict(self._instantiated)

        elif frozen:
            params = {
                n: p.value for n, p in self._parameters.items() if isinstance(p, Frozen)
            }

        else:
            params = dict(self._parameters)

        # recursively add sub-pipeline parameters
        for pipeline_name, pipeline in self._pipelines.items():
            pipeline_params = pipeline._flattened_parameters(
                frozen=frozen, instantiated=instantiated
            )
            for name, value in pipeline_params.items():
                params[f"{pipeline_name}>{name}"] = value

        return params

    def _flatten(self, nested_params: dict) -> dict:
        """Convert nested dictionary to flattened dictionary

        For instance, a nested dictionary like this one:

            ~~~~~~~~~~~~~~~~~~~~~
            param: value1
            pipeline:
                param: value2
                subpipeline:
                    param: value3
            ~~~~~~~~~~~~~~~~~~~~~

        becomes the following flattened dictionary:

            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            param                       : value1
            pipeline>param              : value2
            pipeline>subpipeline>param  : value3
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Parameter
        ---------
        nested_params : `dict`

        Returns
        -------
        flattened_params : `dict`
        """
        flattened_params = {}
        for name, value in nested_params.items():
            if isinstance(value, dict):
                for subname, subvalue in self._flatten(value).items():
                    flattened_params[f"{name}>{subname}"] = subvalue
            else:
                flattened_params[name] = value
        return flattened_params

    def _unflatten(self, flattened_params: dict) -> dict:
        """Convert flattened dictionary to nested dictionary

        For instance, a flattened dictionary like this one:

            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            param                       : value1
            pipeline>param              : value2
            pipeline>subpipeline>param  : value3
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        becomes the following nested dictionary:

            ~~~~~~~~~~~~~~~~~~~~~
            param: value1
            pipeline:
                param: value2
                subpipeline:
                    param: value3
            ~~~~~~~~~~~~~~~~~~~~~

        Parameter
        ---------
        flattened_params : `dict`

        Returns
        -------
        nested_params : `dict`
        """

        nested_params = {}

        pipeline_params = {name: {} for name in self._pipelines}
        for name, value in flattened_params.items():
            # if name contains has multipe ">"-separated tokens
            # it means that it is a sub-pipeline parameter
            tokens = name.split(">")
            if len(tokens) > 1:
                # read sub-pipeline name
                pipeline_name = tokens[0]
                # read parameter name
                param_name = ">".join(tokens[1:])
                # update sub-pipeline flattened dictionary
                pipeline_params[pipeline_name][param_name] = value

            # otherwise, it is an actual parameter of this pipeline
            else:
                # store it as such
                nested_params[name] = value

        # recursively unflatten sub-pipeline flattened dictionary
        for name, pipeline in self._pipelines.items():
            nested_params[name] = pipeline._unflatten(pipeline_params[name])

        return nested_params

    def parameters(
        self,
        trial: Trial | None = None,
        frozen: bool | None = False,
        instantiated: bool | None = False,
    ) -> dict:
        """Returns nested dictionary of (optionnaly instantiated) parameters.

        For a pipeline with one `param`, one sub-pipeline with its own param
        and its own sub-pipeline, it will returns something like:

        ~~~~~~~~~~~~~~~~~~~~~
        param: value1
        pipeline:
            param: value2
            subpipeline:
                param: value3
        ~~~~~~~~~~~~~~~~~~~~~

        Parameter
        ---------
        trial : `Trial`, optional
            When provided, use trial to suggest new parameter values
            and return them.
        frozen : `bool`, optional
            Return frozen parameter value
        instantiated : `bool`, optional
            Return instantiated parameter values.

        Returns
        -------
        params : `dict`
            Nested dictionary of parameters. See above for the actual format.
        """

        if (instantiated or frozen) and trial is not None:
            msg = "One must choose between `trial`, `instantiated`, or `frozen`"
            raise ValueError(msg)

        # get flattened dictionary of uninstantiated parameters
        params = self._flattened_parameters(frozen=frozen, instantiated=instantiated)

        if trial is not None:
            # use provided `trial` to suggest values for parameters
            params = {name: param(name, trial) for name, param in params.items()}

        # un-flatten flattened dictionary
        return self._unflatten(params)

    def initialize(self):
        """Instantiate root pipeline with current set of parameters"""

    def freeze(self, params: dict) -> "Pipeline":
        """Recursively freeze pipeline parameters

        Parameters
        ----------
        params : `dict`
            Nested dictionary of parameters.

        Returns
        -------
        self : `Pipeline`
            Pipeline.
        """

        # imported here to avoid circular imports
        from .parameter import Frozen

        for name, value in params.items():
            # recursively freeze sub-pipelines parameters
            if name in self._pipelines:
                if not isinstance(value, dict):
                    msg = (
                        f"only parameters of '{name}' pipeline can "
                        f"be frozen (not the whole pipeline)"
                    )
                    raise ValueError(msg)
                self._pipelines[name].freeze(value)
                continue

            # instantiate parameter value
            if name in self._parameters:
                setattr(self, name, Frozen(value))
                continue

            msg = f"parameter '{name}' does not exist"
            raise ValueError(msg)

        return self

    def instantiate(self, params: dict) -> "Pipeline":
        """Recursively instantiate all pipelines

        Parameters
        ----------
        params : `dict`
            Nested dictionary of parameters.

        Returns
        -------
        self : `Pipeline`
            Instantiated pipeline.
        """

        # imported here to avoid circular imports
        from .parameter import Frozen

        for name, value in params.items():
            # recursively call `instantiate` with sub-pipelines
            if name in self._pipelines:
                if not isinstance(value, dict):
                    msg = (
                        f"only parameters of '{name}' pipeline can "
                        f"be instantiated (not the whole pipeline)"
                    )
                    raise ValueError(msg)
                self._pipelines[name].instantiate(value)
                continue

            # instantiate parameter value
            if name in self._parameters:
                param = getattr(self, name)
                # overwrite provided value of frozen parameters
                if isinstance(param, Frozen) and param.value != value:
                    msg = (
                        f"Parameter '{name}' is frozen: using its frozen value "
                        f"({param.value}) instead of the one provided ({value})."
                    )
                    warnings.warn(msg)
                    value = param.value
                setattr(self, name, value)
                continue

            msg = f"parameter '{name}' does not exist"
            raise ValueError(msg)

        self.initialize()

        return self

    @property
    def instantiated(self):
        """Whether pipeline has been instantiated (and therefore can be applied)"""
        parameters = set(self._flatten(self.parameters()))
        instantiated = set(self._flatten(self.parameters(instantiated=True)))
        return parameters == instantiated

    def dump_params(
        self,
        params_yml: Path,
        params: dict | None = None,
        loss: float | None = None,
    ) -> str:
        """Dump parameters to disk

        Parameters
        ----------
        params_yml : `Path`
            Path to YAML file.
        params : `dict`, optional
            Nested Parameters. Defaults to pipeline current parameters.
        loss : `float`, optional
            Loss value. Defaults to not write loss to file.

        Returns
        -------
        content : `str`
            Content written in `param_yml`.
        """
        # use instantiated parameters when `params` is not provided
        if params is None:
            params = self.parameters(instantiated=True)

        content = {"params": params}
        if loss is not None:
            content["loss"] = loss

        # format as valid YAML
        content_yml = yaml.dump(content, default_flow_style=False)

        # (safely) dump YAML content
        with FileLock(params_yml.with_suffix(".lock")):
            with open(params_yml, mode="w") as fp:
                fp.write(content_yml)

        return content_yml

    def load_params(self, params_yml: Path) -> "Pipeline":
        """Instantiate pipeline using parameters from disk

        Parameters
        ----------
        param_yml : `Path`
            Path to YAML file.

        Returns
        -------
        self : `Pipeline`
            Instantiated pipeline

        """

        with open(params_yml, mode="r") as fp:
            params = yaml.load(fp, Loader=yaml.SafeLoader)
        return self.instantiate(params["params"])

    def __call__(self, input: PipelineInput) -> PipelineOutput:
        """Apply pipeline on input and return its output"""
        raise NotImplementedError

    def get_metric(
        self,
    ) -> Union[
        "pyannote.metrics.base.BaseMetric",
        Sequence["pyannote.metrics.base.BaseMetric"],
    ]:
        """Return one or several new metrics (from pyannote.metrics)

        When this method is implemented, the returned metric or metrics are
        used as a replacement for the loss method below. Returning several
        metrics enables multi-objective optimization and requires
        :meth:`get_direction` to return the same number of directions.

        Returns
        -------
        metric : `pyannote.metrics.base.BaseMetric` or sequence
        """
        raise NotImplementedError()

    def get_direction(self) -> Directions:
        """Return optimization direction for each objective.

        Returns
        -------
        direction : {"minimize", "maximize"} or sequence
            Optimization direction. Return one direction per metric or loss
            value to enable multi-objective optimization.
        """
        return "minimize"

    def loss(
        self, input: PipelineInput, output: PipelineOutput
    ) -> float | Sequence[float]:
        """Compute one or several losses for given input/output pair

        Parameters
        ----------
        input : object
            Pipeline input.
        output : object
            Pipeline output

        Returns
        -------
        loss : `float` or sequence
            Loss value, or one value per objective for multi-objective
            optimization.
        """
        raise NotImplementedError()

    @property
    def write_format(self):
        return "rttm"

    def write(self, file: TextIO, output: PipelineOutput):
        """Write pipeline output to file

        Parameters
        ----------
        file : file object
        output : object
            Pipeline output
        """

        return getattr(self, f"write_{self.write_format}")(file, output)

    def write_rttm(self, file: TextIO, output: Timeline | Annotation):
        """Write pipeline output to "rttm" file

        Parameters
        ----------
        file : file object
        output : `pyannote.core.Timeline` or `pyannote.core.Annotation`
            Pipeline output
        """

        if isinstance(output, Timeline):
            output = output.to_annotation(generator="string")

        if isinstance(output, Annotation):
            for s, t, l in output.itertracks(yield_label=True):
                line = (
                    f"SPEAKER {output.uri} 1 {s.start:.3f} {s.duration:.3f} "
                    f"<NA> <NA> {l} <NA> <NA>\n"
                )
                file.write(line)
            return

        msg = (
            f'Dumping {output.__class__.__name__} instances to "rttm" files '
            f"is not supported."
        )
        raise NotImplementedError(msg)

    def write_txt(self, file: TextIO, output: Timeline | Annotation):
        """Write pipeline output to "txt" file

        Parameters
        ----------
        file : file object
        output : `pyannote.core.Timeline` or `pyannote.core.Annotation`
            Pipeline output
        """

        if isinstance(output, Timeline):
            for s in output:
                line = f"{output.uri} {s.start:.3f} {s.end:.3f}\n"
                file.write(line)
            return

        if isinstance(output, Annotation):
            for s, t, l in output.itertracks(yield_label=True):
                line = f"{output.uri} {s.start:.3f} {s.end:.3f} {t} {l}\n"
                file.write(line)
            return

        msg = (
            f'Dumping {output.__class__.__name__} instances to "txt" files '
            f"is not supported."
        )
        raise NotImplementedError(msg)
