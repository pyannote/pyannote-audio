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
from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Text, Union

import torch
import torch.nn as nn
import yaml
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder, ProtocolFile
from pyannote.pipeline import Pipeline as _Pipeline

from pyannote.audio import Audio, __version__
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR, Model
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.audio.utils.version import check_version

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

            try:
                config_yml = hf_hub_download(
                    model_id,
                    PIPELINE_PARAMS_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                )

            except RepositoryNotFoundError:
                print(
                    f"""
Could not download '{model_id}' pipeline.
It might be because the pipeline is private or gated so make
sure to authenticate. Visit https://hf.co/settings/tokens to
create your access token and retry with:

   >>> Pipeline.from_pretrained('{model_id}',
   ...                          use_auth_token=YOUR_AUTH_TOKEN)

If this still does not work, it might be because the pipeline is gated:
visit https://hf.co/{model_id} to accept the user conditions."""
                )
                return None

        with open(config_yml, "r") as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

        if "version" in config:
            check_version(
                "pyannote.audio", config["version"], __version__, what="Pipeline"
            )

        # initialize pipeline
        pipeline_name = config["pipeline"]["name"]
        Klass = get_class_by_name(
            pipeline_name, default_module_name="pyannote.pipeline.blocks"
        )
        params = config["pipeline"].get("params", {})
        params.setdefault("use_auth_token", use_auth_token)

        pipeline = Klass(**params)

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

    def save_pretrained(
        self,
        dir,
        save_checkpoints=False,
    ):

        """save pipeline config and model checkpoints to a specific directory:

        Args:
            dir (str): Path directory to save the pipeline and checkpoints
        """

        dir = Path(dir)

        config = {
            "pipeline": {
                "name": str(type(self)).split("'")[1],
                "params": {
                    "clustering": self.klustering,
                    "embedding": self.embedding,
                    "embedding_batch_size": self.embedding_batch_size,
                    "embedding_exclude_overlap": self.embedding_exclude_overlap,
                    "segmentation": self.segmentation,
                    "segmentation_batch_size": self.segmentation_batch_size,
                },
            },
            "params": {
                "clustering": {
                    "method": self._pipelines["clustering"]._instantiated["method"],
                    "min_cluster_size": self._pipelines["clustering"]._instantiated[
                        "min_cluster_size"
                    ],
                    "threshold": self._pipelines["clustering"]._instantiated[
                        "threshold"
                    ],
                },
                "segmentation": {
                    "min_duration_off": self._pipelines["segmentation"]._instantiated[
                        "min_duration_off"
                    ]
                },
            },
            "version": "3.1",
            "checkpoints": save_checkpoints,
        }

        if save_checkpoints:
            seg_path = os.path.join(dir, "segmentation")
            embed_path = os.path.join(dir, "embedding")
            os.makedirs(seg_path, exist_ok=True)
            os.makedirs(embed_path, exist_ok=True)
            self._segmentation.model.save_pretrained(seg_path, model_type="PyanNet")
            self._embedding.model_.save_pretrained(
                embed_path, model_type="WeSpeakerResNet34"
            )

        with open(dir / "config.yaml", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def push_to_hub(
        self,
        repo_id: str,
        save_checkpoints: bool = False,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ):
        """
        Upload the pyannote pipeline config file to the 🤗 Model Hub.

        Important remark:
        --> For now, both push_to_hub and from_pretrained use segmentation and embedding models from the Hub that are loaded using a config file.
        In particular, push_to_hub can't be used (for now) to push a custom pipeline with locally modified models.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            use_auth_token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.
            cache_dir: Path or str, optional
                Path to config cache directory. Defauorch/pyannote" when unset.
        """

        api = HfApi()

        _ = api.create_repo(
            repo_id,
            private=private,
            token=use_auth_token,
            exist_ok=True,
            repo_type="model",
        )

        # Load the pyannote/speaker-diarization-3.1 pipeline config:

        with TemporaryDirectory() as tmpdir:

            self.save_pretrained(tmpdir, save_checkpoints)

            pipeline_card = create_and_tag_pipeline_card(
                repo_id,
                tags,
                use_auth_token=use_auth_token,
            )
            pipeline_card.save(os.path.join(tmpdir, "README.md"))

            return api.upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                use_auth_token=use_auth_token,
                repo_type="model",
                commit_message=commit_message,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )

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
    def setup_hook(file: AudioFile, hook: Optional[Callable] = None) -> Callable:
        def noop(*args, **kwargs):
            return

        return partial(hook or noop, file=file)

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

        if hasattr(self, "preprocessors"):
            file = ProtocolFile(file, lazy=self.preprocessors)

        return self.apply(file, **kwargs)

    def to(self, device: torch.device):
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


def create_and_tag_pipeline_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    use_auth_token: Optional[str] = None,
):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of optional tags to add in the model card
        use_auth_token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
    """

    tags = [] if tags is None else tags

    base_tags = [
        "pyannote",
        "pyannote.audio",
        "pyannote-audio-pipeline",
        "audio",
        "voice",
        "speech",
        "speaker",
        "speaker-diarization",
        "speaker-change-detection",
        "voice-activity-detection",
        "overlapped-speech-detection",
        "automatic-speech-recognition",
    ]
    tags += base_tags

    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id, token=use_auth_token)
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a pyannote pipeline that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(
            tags=[] if tags is None else tags, library_name="pyannote"
        )
        model_card = ModelCard.from_template(
            card_data, model_description=model_description
        )

    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    model_card.data.licence = "mit"

    model_card.text = "This is the model card of a pyannote pipeline that has been pushed on the Hub. This model card has been automatically generated."

    return model_card
