# MIT License
#
# Copyright (c) 2020-2025 CNRS
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

import io
import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import lightning
import torch
import torch.nn as nn
import torch.optim
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import (
    Problem,
    Specifications,
    Task,
    UnknownSpecificationsError,
)
from pyannote.audio.telemetry import track_model_init
from pyannote.audio.utils.dependencies import check_dependencies
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.core import SlidingWindow
from torch.utils.data import DataLoader


# NOTE: needed to backward compatibility to load models trained before pyannote.audio 3.x
class Introspection:
    pass


@dataclass
class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow


class Model(lightning.LightningModule):
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
        task: Task | None = None,
    ):
        super().__init__()

        assert num_channels == 1, (
            "Only mono audio is supported for now (num_channels = 1)"
        )

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
        self._task = task

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    @property
    def specifications(self) -> Specifications | tuple[Specifications]:
        if self.task is None:
            try:
                specifications = self._specifications

            except AttributeError as e:
                raise UnknownSpecificationsError(
                    "Model specifications are not available because it has not been assigned a task yet. "
                    "Use `model.task = ...` to assign a task to the model."
                ) from e

        else:
            specifications = self.task.specifications

        return specifications

    @specifications.setter
    def specifications(self, specifications: Specifications | tuple[Specifications]):
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

    def __example_input_array(self, duration: float | None = None) -> torch.Tensor:
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
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.hparams.sample_rate,
            duration=receptive_field_size / self.hparams.sample_rate,
            step=receptive_field_step / self.hparams.sample_rate,
        )

    def prepare_data(self):
        self.task.prepare_data()

    def setup(self, stage=None):
        if stage == "fit":
            # let the task know about the trainer (e.g for broadcasting
            # cache path between multi-GPU training processes).
            self.task.trainer = self.trainer

        # setup the task if defined (only on training and validation stages,
        # but not for basic inference)
        if self.task:
            self.task.setup(stage)

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
        if self.task:
            # let task know about the model
            self.task.model = self
            # setup custom loss function
            self.task.setup_loss_func()
            # setup custom validation metrics
            self.task.setup_validation_metric()

        # list of layers after adding task-dependent layers
        after = set((name, id(module)) for name, module in self.named_modules())

        # list of task-dependent layers
        self.task_dependent = list(name for name, _ in after - before)

    def on_save_checkpoint(self, checkpoint):
        # put everything pyannote.audio-specific under pyannote.audio
        # to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "pyannote.audio": __version__,
            },
            "architecture": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
            "specifications": self.specifications,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        self.specifications = checkpoint["pyannote.audio"]["specifications"]

        # add task-dependent (e.g. final classifier) layers
        self.setup()

    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor]:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> nn.Module | tuple[nn.Module]:
        """Guess default activation function according to task specification

            * sigmoid for binary classification
            * log-softmax for regular multi-class classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : (tuple of) nn.Module
            Activation.
        """

        def __default_activation(
            specifications: Specifications | None = None,
        ) -> nn.Module:
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

    def __up_to(self, module_name: str, requires_grad: bool = False) -> list[str]:
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

    def freeze_up_to(self, module_name: str) -> list[str]:
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

    def unfreeze_up_to(self, module_name: str) -> list[str]:
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
        modules: list[str] | str,
        recurse: bool = True,
        requires_grad: bool = False,
    ) -> list[str]:
        """Helper function for freeze_by_name and unfreeze_by_name"""

        updated_modules = list()

        # Force modules to be a list
        if isinstance(modules, str):
            modules = [modules]

        for name in modules:
            module = getattr(self, name)

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
        modules: str | list[str],
        recurse: bool = True,
    ) -> list[str]:
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
        modules: list[str] | str,
        recurse: bool = True,
    ) -> list[str]:
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
        checkpoint: Path | str | io.BytesIO,
        map_location=None,
        strict: bool = True,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: Path | str | None = None,
        **kwargs,
    ) -> Optional["Model"]:
        """Load pretrained model

        Parameters
        ----------
        checkpoint : Path, str, or byte buffer
            Model checkpoint, provided as one of the following:
            * path to a local `pytorch_model.bin` model checkpoint
            * path to a local directory containing such a file
            * identifier of a model on Huggingface hub
            * pre-loaded model checkpoint as a byte buffer
        map_location: optional
            Same role as in torch.load().
            Defaults to `lambda storage, loc: storage`.
        strict : bool, optional
            Whether to strictly enforce that the keys in checkpoint match
            the keys returned by this module’s state dict. Defaults to True.
        subfolder : str, optional
            Folder inside the hf.co model repo.
        revision : str, optional
            Revision when loading from the huggingface.co model hub.
        token : str or bool, optional
            Token to be used for the download.
        cache_dir: Path or str, optional
            Path to the folder where cached files are stored.
        kwargs: optional
            Any extra keyword args needed to init the model.
            Can also be used to override saved hyperparameter values.

        Returns
        -------
        model : Model
            Model

        See also
        --------
        `huggingface_hub.hf_hub_download`

        """
        if isinstance(checkpoint, io.BytesIO) or os.path.isfile(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")

            # if checkpoint is a BytesIO object or a file path, use it as is
            path_to_model_checkpoint = checkpoint
            otel_origin: str = "local"

        # if checkpoint is a directory, look for the model checkpoint
        # inside this directory (or inside a subfolder if specified)
        elif os.path.isdir(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")

            if subfolder:
                path_to_model_checkpoint = (
                    Path(checkpoint) / subfolder / AssetFileName.Model.value
                )
            else:
                path_to_model_checkpoint = Path(checkpoint) / AssetFileName.Model.value
            otel_origin: str = "local"

        # otherwise, assume that the checkpoint is hosted on HF model hub
        else:
            checkpoint = str(checkpoint)
            if "@" in checkpoint:
                raise ValueError(
                    "Revisions must be passed with `revision` keyword argument."
                )

            path_to_model_checkpoint = download_from_hf_hub(
                checkpoint,
                AssetFileName.Model,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            if path_to_model_checkpoint is None:
                return None

            otel_origin: str = (
                checkpoint
                if checkpoint.lower().startswith(("pyannote/", "pyannoteai/"))
                else "huggingface"
            )

        if map_location is None:

            def default_map_location(storage, loc):
                return storage

            map_location = default_map_location

        # load checkpoint using lightning
        loaded_checkpoint = pl_load(path_to_model_checkpoint, map_location=map_location)

        # check that the checkpoint is compatible with the current version
        versions = loaded_checkpoint["pyannote.audio"]["versions"]
        check_dependencies({"pyannote.audio": versions["pyannote.audio"]}, "Model")

        # obtain model class from the checkpoint
        module_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["module"]
        module = import_module(module_name)
        class_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["class"]
        Klass = getattr(module, class_name)

        try:
            # if checkpoint is a BytesIO object, seek to the beginning so we can load it again
            if isinstance(path_to_model_checkpoint, io.BytesIO):
                path_to_model_checkpoint.seek(0)

            model = Klass.load_from_checkpoint(
                path_to_model_checkpoint,
                map_location=map_location,
                strict=strict,
                **kwargs,
            )
        except RuntimeError as e:
            if "loss_func" in str(e):
                # if checkpoint is a BytesIO object, seek to the beginning so we can load it again
                if isinstance(path_to_model_checkpoint, io.BytesIO):
                    path_to_model_checkpoint.seek(0)

                msg = (
                    "Model has been trained with a task-dependent loss function. "
                    "Set 'strict' to False to load the model without its loss function "
                    "and prevent this warning from appearing. "
                )
                warnings.warn(msg)
                model = Klass.load_from_checkpoint(
                    path_to_model_checkpoint,
                    map_location=map_location,
                    strict=False,
                    **kwargs,
                )
                return model

            raise e

        # save model origin (HF, local, etc) as attribute for telemetry purposes
        model._otel_origin = otel_origin
        track_model_init(model)

        return model
