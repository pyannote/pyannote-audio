# MIT License
#
# Copyright (c) 2021-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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

from __future__ import annotations
import os
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from pyannote.audio import Audio
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.telemetry import track_pipeline_apply, track_pipeline_init
from pyannote.audio.utils.dependencies import check_dependencies
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, ProtocolFile
from pyannote.pipeline import Pipeline as _Pipeline


def expand_subfolders(
    config,
    model_id: str | None = None,
    parent_revision: str | None = None,
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> None:
    """Expand $model subfolders in config

    Processes `config` dictionary recursively and replaces "$model/{subfolder}"
    values with {"checkpoint": model_id,
                 "subfolder": {subfolder},
                 "token": token}

    Parameters
    ----------
    config : dict
    model_id : str, optional
        Model identifier when loading from the huggingface.co model hub.
    parent_revision : str, optional
        Revision when loading from the huggingface.co model hub.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.
    """

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("$model/"):
                subfolder = "/".join(value.split("/")[1:])

                # if subfolder contains '@', split it to get revision
                if "@" in subfolder:
                    subfolder, revision = subfolder.split("@")
                # otherwise, use parent revision if any
                else:
                    revision = parent_revision

                config[key] = {
                    "checkpoint": model_id,
                    "revision": revision,
                    "subfolder": subfolder,
                    "token": token,
                    "cache_dir": cache_dir,
                }
            else:
                expand_subfolders(
                    value,
                    model_id,
                    parent_revision=parent_revision,
                    token=token,
                    cache_dir=cache_dir,
                )

    elif isinstance(config, list):
        for idx, value in enumerate(config):
            if isinstance(value, str) and value.startswith("$model/"):
                subfolder = "/".join(value.split("/")[1:])

                # if subfolder contains '@', split it to get revision
                if "@" in subfolder:
                    subfolder, revision = subfolder.split("@")
                # otherwise, use parent revision if any
                else:
                    revision = parent_revision

                config[idx] = {
                    "checkpoint": model_id,
                    "revision": revision,
                    "subfolder": subfolder,
                    "token": token,
                    "cache_dir": cache_dir,
                }

            else:
                expand_subfolders(
                    value,
                    model_id,
                    parent_revision=parent_revision,
                    token=token,
                    cache_dir=cache_dir,
                )


class Pipeline(_Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | dict,
        revision: str | None = None,
        hparams_file: str | Path | None = None,
        token: str | bool | None = None,
        cache_dir: Path | str | None = None,
    ) -> Optional["Pipeline"]:
        """Load pretrained pipeline

        Parameters
        ----------
        checkpoint : str
            Pipeline checkpoint, provided as one of the following:
            * path to a local `config.yaml` pipeline checkpoint
            * path to a local directory containing such a file
            * identifier (str) of a pipeline on huggingface.co model hub
            * dictionary containing the actual content of a config file
        revision : str, optional
            Revision when loading from the huggingface.co model hub.
        hparams_file: Path or str, optional
        token : str or bool, optional
            Token to be used for the download.
        cache_dir: Path or str, optional
            Path to the folder where cached files are stored.
        """

        # if checkpoint is a dict, assume it is the actual content of
        # a config file
        if isinstance(checkpoint, dict):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")
            model_id = Path.cwd()
            config = checkpoint
            otel_origin: str = "local"

        # if checkpoint is a directory, look for the pipeline checkpoint
        # inside this directory
        elif os.path.isdir(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")
            model_id = Path(checkpoint)
            config_yml = model_id / AssetFileName.Pipeline.value
            otel_origin: str = "local"

        # if checkpoint is a file, assume it is the pipeline checkpoint
        elif os.path.isfile(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")
            model_id = Path(checkpoint).parent
            config_yml = checkpoint
            otel_origin: str = "local"

        # otherwise, assume that the checkpoint is hosted on HF model hub
        else:
            model_id = str(checkpoint)

            if "@" in model_id:
                raise ValueError(
                    "Revisions must be passed with `revision` keyword argument."
                )

            config_yml = download_from_hf_hub(
                model_id,
                AssetFileName.Pipeline,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            if config_yml is None:
                return None

            otel_origin: str = (
                model_id
                if model_id.lower().startswith(("pyannote/", "pyannoteai/"))
                else "huggingface"
            )

        if not isinstance(checkpoint, dict):
            with open(config_yml, "r") as fp:
                config = yaml.load(fp, Loader=yaml.SafeLoader)

        # expand $model/{subfolder}-like entries in config
        expand_subfolders(
            config,
            model_id,
            parent_revision=revision,
            token=token,
            cache_dir=cache_dir,
        )

        # before 4.x, pyannote.audio pipeline was using "version" key to
        # specify the version of pyannote.audio used to train the pipeline
        if "version" in config:
            config["dependencies"] = {"pyannote.audio": config["version"]}
            del config["version"]

        # check that dependencies are available (in their required version)
        dependencies: dict[str, str] = config.get("dependencies", dict())
        check_dependencies(dependencies, "Pipeline")

        # initialize pipeline
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        params = config["pipeline"].get("params", {})
        params.setdefault("token", token)
        params.setdefault("cache_dir", cache_dir)
        pipeline = Klass(**params)

        # save pipeline origin (HF, local, etc) and class name as attributes for telemetry purposes
        pipeline._otel_origin = otel_origin
        pipeline._otel_name = pipeline_name
        track_pipeline_init(pipeline)

        # freeze  parameters
        if "freeze" in config:
            params = config["freeze"]
            pipeline.freeze(params)

        if "params" in config:
            pipeline.instantiate(config["params"])

        if hparams_file is not None:
            pipeline.load_params(hparams_file)

        if "preprocessors" in config:
            preprocessors = {}
            for key, preprocessor in config.get("preprocessors", {}).items():
                # preprocessors:
                #    key:
                #       name: package.module.ClassName
                #       params:
                #          param1: value1
                #          param2: value2
                if isinstance(preprocessor, dict):
                    Klass = get_class_by_name(
                        preprocessor["name"], default_module_name="pyannote.audio"
                    )
                    params = preprocessor.get("params", {})
                    preprocessors[key] = Klass(**params)
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

            pipeline.preprocessors = preprocessors

        # send pipeline to specified device
        if "device" in config:
            device = torch.device(config["device"])
            try:
                pipeline.to(device)
            except RuntimeError as e:
                print(e)

        return pipeline

    def __init__(self):
        super().__init__()
        self._models: Dict[str, Model] = OrderedDict()
        self._inferences: Dict[str, BaseInference] = OrderedDict()

    def __getattr__(self, name):
        """(Advanced) attribute getter

        Adds support for Model and Inference attributes,
        which are iterated over by Pipeline.to() method.

        See pyannote.pipeline.Pipeline.__getattr__.
        """

        if "_models" in self.__dict__:
            _models = self.__dict__["_models"]
            if name in _models:
                return _models[name]

        if "_inferences" in self.__dict__:
            _inferences = self.__dict__["_inferences"]
            if name in _inferences:
                return _inferences[name]

        return super().__getattr__(name)

    def __setattr__(self, name, value):
        """(Advanced) attribute setter

        Adds support for Model and Inference attributes,
        which are iterated over by Pipeline.to() method.

        See pyannote.pipeline.Pipeline.__setattr__.
        """

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        _parameters = self.__dict__.get("_parameters")
        _instantiated = self.__dict__.get("_instantiated")
        _pipelines = self.__dict__.get("_pipelines")
        _models = self.__dict__.get("_models")
        _inferences = self.__dict__.get("_inferences")

        if isinstance(value, nn.Module):
            if _models is None:
                msg = "cannot assign models before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(
                self.__dict__, _inferences, _parameters, _instantiated, _pipelines
            )
            _models[name] = value
            return

        if isinstance(value, BaseInference):
            if _inferences is None:
                msg = "cannot assign inferences before Pipeline.__init__() call"
                raise AttributeError(msg)
            remove_from(self.__dict__, _models, _parameters, _instantiated, _pipelines)
            _inferences[name] = value
            return

        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._models:
            del self._models[name]

        elif name in self._inferences:
            del self._inferences[name]

        else:
            super().__delattr__(name)

    @staticmethod
    def setup_hook(file: AudioFile, hook: Callable | None = None) -> Callable:
        def noop(*args, **kwargs):
            return

        return partial(hook or noop, file=file)

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self) -> List | Iterator:
        """Classes returned by the pipeline

        Returns
        -------
        classes : list of string or string iterator
            Finite list of strings when classes are known in advance
            (e.g. ["MALE", "FEMALE"] for gender classification), or
            infinite string iterator when they depend on the file
            (e.g. "SPEAKER_00", "SPEAKER_01", ... for speaker diarization)

        Usage
        -----
        >>> from collections.abc import Iterator
        >>> classes = pipeline.classes()
        >>> if isinstance(classes, Iterator):  # classes depend on the input file
        >>> if isinstance(classes, list):      # classes are known in advance

        """
        raise NotImplementedError()

    def __call__(self, file: AudioFile, preload: bool = False, **kwargs):
        """Validate file, (optionally) load it in memory, then process it

        Parameters
        ----------
        file : AudioFile
            File to process
        preload : bool, optional
            Whether to preload waveform before applying the pipeline.
        kwargs : keyword arguments, optional
            Additional keyword arguments passed to `self.apply(...)`

        Returns
        -------
        output : Any
            Whatever `self.apply(...)` returns
        """
        fix_reproducibility(getattr(self, "device", torch.device("cpu")))

        if not self.instantiated:
            # instantiate with default parameters when available
            try:
                default_parameters = self.default_parameters()
            except NotImplementedError:
                raise RuntimeError(
                    "A pipeline must be instantiated with `pipeline.instantiate(parameters)` before it can be applied."
                )

            try:
                self.instantiate(default_parameters)
            except ValueError:
                raise RuntimeError(
                    "A pipeline must be instantiated with `pipeline.instantiate(paramaters)` before it can be applied. "
                    "Tried to use parameters provided by `pipeline.default_parameters()` but those are not compatible. "
                )

            warnings.warn(
                f"The pipeline has been automatically instantiated with {default_parameters}."
            )

        file = Audio.validate_file(file)

        # check if the instance has preprocessors and wrap the file if so
        if hasattr(self, "preprocessors"):
            file = ProtocolFile(file, lazy=self.preprocessors)

        # pre-load the audio in memory if requested
        if preload:
            # raise error if `waveform`` is already in memory (or will be via a preprocessor)
            if (
                "waveform" in getattr(self, "preprocessors", dict())
                or "waveform" in file
            ):
                raise ValueError(
                    "Cannot preload audio: `waveform` key is already available or will be via a preprocessor."
                )

            # load waveform in memory (and keep track of its original sample rate)
            file["waveform"], file["sample_rate"] = Audio()(file)

            # the above line already took care of channel selection,
            # therefore we remove the `channel` key from the file
            file.pop("channel", None)

        # send file duration to telemetry as well as
        # requested number of speakers in case of diarization
        track_pipeline_apply(self, file, **kwargs)

        return self.apply(file, **kwargs)

    def to(self, device: torch.device) -> Pipeline:
        """Send pipeline to `device`"""

        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be an instance of `torch.device`, got `{type(device).__name__}`"
            )

        for _, pipeline in self._pipelines.items():
            if hasattr(pipeline, "to"):
                _ = pipeline.to(device)

        for _, model in self._models.items():
            _ = model.to(device)

        for _, inference in self._inferences.items():
            _ = inference.to(device)

        self.device = device

        return self

    def cuda(self, device: torch.device | int | None = None) -> Pipeline:
        """Send pipeline to (optionally specified) cuda device"""
        if device is None:
            return self.to(torch.device("cuda"))
        elif isinstance(device, int):
            return self.to(torch.device("cuda", device))
        else:
            if device.type != "cuda":
                raise ValueError("Expected CUDA device. Use `Pipeline.to(device)` for other devices.")
            return self.to(device)
