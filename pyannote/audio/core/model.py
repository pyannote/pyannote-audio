# MIT License
#
# Copyright (c) 2020- CNRS
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
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union
from urllib.parse import urlparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from lightning_fabric.utilities.cloud_io import _load as pl_load
from pyannote.core import SlidingWindow
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader

from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import (
    Problem,
    Resolution,
    Specifications,
    Task,
    UnknownSpecificationsError,
)
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.version import check_version

CACHE_DIR = os.getenv(
    "PYANNOTE_CACHE",
    os.path.expanduser("~/.cache/torch/pyannote"),
)
HF_PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
HF_LIGHTNING_CONFIG_NAME = "config.yaml"


# NOTE: needed to backward compatibility to load models trained before pyannote.audio 3.x
class Introspection:
    pass


@dataclass
class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow


class Model(pl.LightningModule):
    """Base model

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task addressed by the model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__()

        assert (
            num_channels == 1
        ), "Only mono audio is supported for now (num_channels = 1)"

        self.save_hyperparameters("sample_rate", "num_channels")

        self.task = task
        self.audio = Audio(sample_rate=self.hparams.sample_rate, mono="downmix")

    @property
    def task(self) -> Task:
        return self._task

    @task.setter
    def task(self, task: Task):
        # reset (cached) properties when task changes
        del self.specifications
        try:
            del self.example_output
        except AttributeError:
            pass
        self._task = task

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    @property
    def specifications(self) -> Union[Specifications, Tuple[Specifications]]:
        if self.task is None:
            try:
                specifications = self._specifications

            except AttributeError as e:
                raise UnknownSpecificationsError(
                    "Model specifications are not available because it has not been assigned a task yet. "
                    "Use `model.task = ...` to assign a task to the model."
                ) from e

        else:
            try:
                specifications = self.task.specifications

            except AttributeError as e:
                raise UnknownSpecificationsError(
                    "Task specifications are not available. This is most likely because they depend on "
                    "the content of the training subset. Use `model.task.setup()` to go over the training "
                    "subset and fix this, or let lightning trainer do that for you in `trainer.fit(model)`."
                ) from e

        return specifications

    @specifications.setter
    def specifications(
        self, specifications: Union[Specifications, Tuple[Specifications]]
    ):
        if not isinstance(specifications, (Specifications, tuple)):
            raise ValueError(
                "Only regular specifications or tuple of specifications are supported."
            )

        durations = set(s.duration for s in specifications)
        if len(durations) > 1:
            raise ValueError("All tasks must share the same (maximum) duration.")

        min_durations = set(s.min_duration for s in specifications)
        if len(min_durations) > 1:
            raise ValueError("All tasks must share the same minimum duration.")

        self._specifications = specifications

    @specifications.deleter
    def specifications(self):
        if hasattr(self, "_specifications"):
            del self._specifications

    def __example_input_array(self, duration: Optional[float] = None) -> torch.Tensor:
        duration = duration or next(iter(self.specifications)).duration
        return torch.randn(
            (
                1,
                self.hparams.num_channels,
                self.audio.get_num_samples(duration),
            ),
            device=self.device,
        )

    @property
    def example_input_array(self) -> torch.Tensor:
        return self.__example_input_array()

    @cached_property
    def example_output(self) -> Union[Output, Tuple[Output]]:
        """Example output"""
        example_input_array = self.__example_input_array()
        with torch.inference_mode():
            example_output = self(example_input_array)

        def __example_output(
            example_output: torch.Tensor,
            specifications: Specifications = None,
        ) -> Output:
            if specifications.resolution == Resolution.FRAME:
                _, num_frames, dimension = example_output.shape
                frame_duration = specifications.duration / num_frames
                frames = SlidingWindow(step=frame_duration, duration=frame_duration)
            else:
                _, dimension = example_output.shape
                num_frames = None
                frames = None

            return Output(
                num_frames=num_frames,
                dimension=dimension,
                frames=frames,
            )

        return map_with_specifications(
            self.specifications, __example_output, example_output
        )

    def setup(self, stage=None):
        if stage == "fit":
            self.task.setup_metadata()

        # list of layers before adding task-dependent layers
        before = set((name, id(module)) for name, module in self.named_modules())

        # add task-dependent layers (e.g. final classification layer)
        # and re-use original weights when compatible

        original_state_dict = self.state_dict()
        self.build()

        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                original_state_dict, strict=False
            )

        except RuntimeError as e:
            if "size mismatch" in str(e):
                msg = (
                    "Model has been trained for a different task. For fine tuning or transfer learning, "
                    "it is recommended to train task-dependent layers for a few epochs "
                    f"before training the whole model: {self.task_dependent}."
                )
                warnings.warn(msg)
            else:
                raise e

        # move layers that were added by build() to same device as the rest of the model
        for name, module in self.named_modules():
            if (name, id(module)) not in before:
                module.to(self.device)

        # add (trainable) loss function (e.g. ArcFace has its own set of trainable weights)
        if stage == "fit":
            # let task know about the model
            self.task.model = self
            # setup custom loss function
            self.task.setup_loss_func()
            # setup custom validation metrics
            self.task.setup_validation_metric()

            # cache for later (and to avoid later CUDA error with multiprocessing)
            _ = self.example_output

        # list of layers after adding task-dependent layers
        after = set((name, id(module)) for name, module in self.named_modules())

        # list of task-dependent layers
        self.task_dependent = list(name for name, _ in after - before)

    def on_save_checkpoint(self, checkpoint):
        # put everything pyannote.audio-specific under pyannote.audio
        # to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "torch": torch.__version__,
                "pyannote.audio": __version__,
            },
            "architecture": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
            "specifications": self.specifications,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        check_version(
            "pyannote.audio",
            checkpoint["pyannote.audio"]["versions"]["pyannote.audio"],
            __version__,
            what="Model",
        )

        check_version(
            "torch",
            checkpoint["pyannote.audio"]["versions"]["torch"],
            torch.__version__,
            what="Model",
        )

        check_version(
            "pytorch-lightning",
            checkpoint["pytorch-lightning_version"],
            pl.__version__,
            what="Model",
        )

        self.specifications = checkpoint["pyannote.audio"]["specifications"]

        # add task-dependent (e.g. final classifier) layers
        self.setup()

    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> Union[nn.Module, Tuple[nn.Module]]:
        """Guess default activation function according to task specification

            * sigmoid for binary classification
            * log-softmax for regular multi-class classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : (tuple of) nn.Module
            Activation.
        """

        def __default_activation(specifications: Specifications = None) -> nn.Module:
            if specifications.problem == Problem.BINARY_CLASSIFICATION:
                return nn.Sigmoid()

            elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
                return nn.LogSoftmax(dim=-1)

            elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
                return nn.Sigmoid()

            else:
                msg = "TODO: implement default activation for other types of problems"
                raise NotImplementedError(msg)

        return map_with_specifications(self.specifications, __default_activation)

    # training data logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def train_dataloader(self) -> DataLoader:
        return self.task.train_dataloader()

    # training step logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def training_step(self, batch, batch_idx):
        return self.task.training_step(batch, batch_idx)

    # validation data logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def val_dataloader(self) -> DataLoader:
        return self.task.val_dataloader()

    # validation logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def validation_step(self, batch, batch_idx):
        return self.task.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def __up_to(self, module_name: Text, requires_grad: bool = False) -> List[Text]:
        """Helper function for freeze_up_to and unfreeze_up_to"""

        tokens = module_name.split(".")
        updated_modules = list()

        for name, module in ModelSummary(self, max_depth=-1).named_modules:
            name_tokens = name.split(".")
            matching_tokens = list(
                token
                for token, other_token in zip(name_tokens, tokens)
                if token == other_token
            )

            # if module is A.a.1 & name is A.a, we do not want to freeze the whole A.a module
            # because it might contain other modules like A.a.2 and A.a.3
            if matching_tokens and len(matching_tokens) == len(tokens) - 1:
                continue

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            module.train(mode=requires_grad)

            updated_modules.append(name)

            #  stop once we reached the requested module
            if module_name == name:
                break

        if module_name not in updated_modules:
            raise ValueError(f"Could not find module {module_name}")

        return updated_modules

    def freeze_up_to(self, module_name: Text) -> List[Text]:
        """Freeze model up to specific module

        Parameters
        ----------
        module_name : str
            Name of module (included) up to which the model will be frozen.

        Returns
        -------
        frozen_modules : list of str
            List of names of frozen modules

        Raises
        ------
        ValueError when requested module does not exist

        Note
        ----
        The order of modules is the one reported by self.summary("full").
        If your model does not follow a sequential structure, you might
        want to use freeze_by_name for more control.
        """
        return self.__up_to(module_name, requires_grad=False)

    def unfreeze_up_to(self, module_name: Text) -> List[Text]:
        """Unfreeze model up to specific module

        Parameters
        ----------
        module_name : str
            Name of module (included) up to which the model will be unfrozen.

        Returns
        -------
        unfrozen_modules : list of str
            List of names of frozen modules

        Raises
        ------
        ValueError when requested module does not exist

        Note
        ----
        The order of modules is the one reported by self.summary("full").
        If your model does not follow a sequential structure, you might
        want to use freeze_by_name for more control.
        """
        return self.__up_to(module_name, requires_grad=True)

    def __by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
        requires_grad: bool = False,
    ) -> List[Text]:
        """Helper function for freeze_by_name and unfreeze_by_name"""

        updated_modules = list()

        # Force modules to be a list
        if isinstance(modules, str):
            modules = [modules]

        for name, module in ModelSummary(self, max_depth=-1).named_modules:
            if name not in modules:
                continue

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            module.train(requires_grad)

            # keep track of updated modules
            updated_modules.append(name)

        missing = list(set(modules) - set(updated_modules))
        if missing:
            raise ValueError(f"Could not find the following modules: {missing}.")

        return updated_modules

    def freeze_by_name(
        self,
        modules: Union[Text, List[Text]],
        recurse: bool = True,
    ) -> List[Text]:
        """Freeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to freeze
        recurse : bool, optional
            If True (default), freezes parameters of these modules and all submodules.
            Otherwise, only freezes parameters that are direct members of these modules.

        Returns
        -------
        frozen_modules: list of str
            Names of frozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(
            modules,
            recurse=recurse,
            requires_grad=False,
        )

    def unfreeze_by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
    ) -> List[Text]:
        """Unfreeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to unfreeze
        recurse : bool, optional
            If True (default), unfreezes parameters of these modules and all submodules.
            Otherwise, only unfreezes parameters that are direct members of these modules.

        Returns
        -------
        unfrozen_modules: list of str
            Names of unfrozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(modules, recurse=recurse, requires_grad=True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text],
        map_location=None,
        hparams_file: Union[Path, Text] = None,
        strict: bool = True,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
        **kwargs,
    ) -> "Model":
        """Load pretrained model

        Parameters
        ----------
        checkpoint : Path or str
            Path to checkpoint, or a remote URL, or a model identifier from
            the huggingface.co model hub.
        map_location: optional
            Same role as in torch.load().
            Defaults to `lambda storage, loc: storage`.
        hparams_file : Path or str, optional
            Path to a .yaml file with hierarchical structure as in this example:
                drop_prob: 0.2
                dataloader:
                    batch_size: 32
            You most likely won’t need this since Lightning will always save the
            hyperparameters to the checkpoint. However, if your checkpoint weights
            do not have the hyperparameters saved, use this method to pass in a .yaml
            file with the hparams you would like to use. These will be converted
            into a dict and passed into your Model for use.
        strict : bool, optional
            Whether to strictly enforce that the keys in checkpoint match
            the keys returned by this module’s state dict. Defaults to True.
        use_auth_token : str, optional
            When loading a private huggingface.co model, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory. Defaults to content of PYANNOTE_CACHE
            environment variable, or "~/.cache/torch/pyannote" when unset.
        kwargs: optional
            Any extra keyword args needed to init the model.
            Can also be used to override saved hyperparameter values.

        Returns
        -------
        model : Model
            Model

        See also
        --------
        torch.load
        """

        # pytorch-lightning expects str, not Path.
        checkpoint = str(checkpoint)
        if hparams_file is not None:
            hparams_file = str(hparams_file)

        # resolve the checkpoint to
        # something that pl will handle
        if os.path.isfile(checkpoint):
            path_for_pl = checkpoint
        elif urlparse(checkpoint).scheme in ("http", "https"):
            path_for_pl = checkpoint
        else:
            # Finally, let's try to find it on Hugging Face model hub
            # e.g. julien-c/voice-activity-detection is a valid model id
            # and  julien-c/voice-activity-detection@main supports specifying a commit/branch/tag.
            if "@" in checkpoint:
                model_id = checkpoint.split("@")[0]
                revision = checkpoint.split("@")[1]
            else:
                model_id = checkpoint
                revision = None

            try:
                path_for_pl = hf_hub_download(
                    model_id,
                    HF_PYTORCH_WEIGHTS_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    # force_download=False,
                    # proxies=None,
                    # etag_timeout=10,
                    # resume_download=False,
                    use_auth_token=use_auth_token,
                    # local_files_only=False,
                    # legacy_cache_layout=False,
                )
            except RepositoryNotFoundError:
                print(
                    f"""
Could not download '{model_id}' model.
It might be because the model is private or gated so make
sure to authenticate. Visit https://hf.co/settings/tokens to
create your access token and retry with:

   >>> Model.from_pretrained('{model_id}',
   ...                       use_auth_token=YOUR_AUTH_TOKEN)

If this still does not work, it might be because the model is gated:
visit https://hf.co/{model_id} to accept the user conditions."""
                )
                return None

            # HACK Huggingface download counters rely on config.yaml
            # HACK Therefore we download config.yaml even though we
            # HACK do not use it. Fails silently in case model does not
            # HACK have a config.yaml file.
            try:
                _ = hf_hub_download(
                    model_id,
                    HF_LIGHTNING_CONFIG_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    # force_download=False,
                    # proxies=None,
                    # etag_timeout=10,
                    # resume_download=False,
                    use_auth_token=use_auth_token,
                    # local_files_only=False,
                    # legacy_cache_layout=False,
                )

            except Exception:
                pass

        if map_location is None:

            def default_map_location(storage, loc):
                return storage

            map_location = default_map_location

        # obtain model class from the checkpoint
        loaded_checkpoint = pl_load(path_for_pl, map_location=map_location)
        module_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["module"]
        module = import_module(module_name)
        class_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["class"]
        Klass = getattr(module, class_name)

        try:
            model = Klass.load_from_checkpoint(
                path_for_pl,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except RuntimeError as e:
            if "loss_func" in str(e):
                msg = (
                    "Model has been trained with a task-dependent loss function. "
                    "Set 'strict' to False to load the model without its loss function "
                    "and prevent this warning from appearing. "
                )
                warnings.warn(msg)
                model = Klass.load_from_checkpoint(
                    path_for_pl,
                    map_location=map_location,
                    hparams_file=hparams_file,
                    strict=False,
                    **kwargs,
                )
                return model

            raise e

        return model
