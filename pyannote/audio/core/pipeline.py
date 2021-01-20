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
from pathlib import Path
from typing import Mapping, Text, Union

import yaml
from huggingface_hub import cached_download, hf_hub_url

from pyannote.audio import __version__
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, ProtocolFile
from pyannote.pipeline import Pipeline as _Pipeline

PIPELINE_PARAMS_NAME = "pipeline.yaml"


class Pipeline(_Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        pipeline_yml: Union[Text, Path],
        params_yml: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
    ) -> "Pipeline":
        """Load pretrained pipeline"""

        pipeline_yml = str(pipeline_yml)

        if os.path.isfile(pipeline_yml):
            config_yml = pipeline_yml

        else:
            if "@" in pipeline_yml:
                model_id = pipeline_yml.split("@")[0]
                revision = pipeline_yml.split("@")[1]
            else:
                model_id = pipeline_yml
                revision = None
            url = hf_hub_url(
                model_id=model_id, filename=PIPELINE_PARAMS_NAME, revision=revision
            )

            config_yml = cached_download(
                url=url,
                library_name="pyannote",
                library_version=__version__,
                cache_dir=CACHE_DIR,
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

        if params_yml is not None:
            pipeline.load_params(params_yml)

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

            pipeline.preprocessors = preprocessors

        return pipeline

    def __call__(self, file: AudioFile):

        if hasattr(self, "preprocessors"):

            if not isinstance(file, Mapping):
                file = {"audio": file}

            file = ProtocolFile(file, lazy=self.preprocessors)

        return super().__call__(file)
