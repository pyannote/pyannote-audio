# MIT License
#
# Copyright (c) 2021 CNRS
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

import os
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Text, Union

import yaml
from huggingface_hub import cached_download, hf_hub_url

from pyannote.audio import Audio, __version__
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, ProtocolFile
from pyannote.pipeline import Pipeline as _Pipeline

PIPELINE_PARAMS_NAME = "config.yaml"


class Pipeline(_Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[Text, Path],
        hparams_file: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ) -> "Pipeline":
        """Load pretrained pipeline

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to pipeline checkpoint, or a remote URL,
            or a pipeline identifier from the huggingface.co model hub.
        hparams_file: Path or str, optional
        use_auth_token : str, optional
            When loading a private huggingface.co pipeline, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory. Defauorch/pyannote" when unset.
        """

        checkpoint_path = str(checkpoint_path)

        if os.path.isfile(checkpoint_path):
            config_yml = checkpoint_path

        else:
            if "@" in checkpoint_path:
                model_id = checkpoint_path.split("@")[0]
                revision = checkpoint_path.split("@")[1]
            else:
                model_id = checkpoint_path
                revision = None
            url = hf_hub_url(model_id, filename=PIPELINE_PARAMS_NAME, revision=revision)

            config_yml = cached_download(
                url=url,
                library_name="pyannote",
                library_version=__version__,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )

        with open(config_yml, "r") as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

        # initialize pipeline
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        pipeline = Klass(**config["pipeline"].get("params", {}))

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

        return pipeline

    @staticmethod
    def setup_hook(file: AudioFile, hook: Optional[Callable] = None) -> Callable:

        if hook is None:

            def hook(*args, **kwargs):
                return

            hook.missing = True
        else:
            hook = partial(hook, file=file)
            hook.missing = False

        return hook

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self) -> Union[List, Iterator]:
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

    def __call__(self, file: AudioFile, **kwargs):
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

        if hasattr(self, "preprocessors"):
            file = ProtocolFile(file, lazy=self.preprocessors)

        return self.apply(file, **kwargs)
