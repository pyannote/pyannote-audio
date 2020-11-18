# MIT License
#
# Copyright (c) 2020 CNRS
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

import warnings
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.cloud_io import load as pl_load
from semver import VersionInfo

from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification


@dataclass
class ModelIntrospection:
    # minimum number of input samples
    min_num_samples: int
    # corresponding minimum number of output frames
    min_num_frames: int
    # number of input samples leading to an increase of number of output frames
    inc_num_samples: int
    # corresponding increase in number of output frames
    inc_num_frames: int
    # output dimension
    dimension: int

    def __call__(self, num_samples: int) -> Tuple[int, int]:
        """Estimate output shape

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames
        dimension : int
            Dimension of output frames

        """

        if num_samples < self.min_num_samples:
            return 0, self.dimension

        return (
            self.min_num_frames
            + self.inc_num_frames
            * ((num_samples - self.min_num_samples + 1) // self.inc_num_samples),
            self.dimension,
        )

    def __len__(self):
        # makes it possible to do something like:
        # multi_task = len(model_introspection) > 1
        # because multi-task introspections are stored as {task_name: introspection} dict
        return 1

    def items(self):
        yield None, self


class Model(pl.LightningModule):
    """Base model

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task addressed by the model. Only provided when training the model.
        A model should be `load_from_checkpoint`-able without a task as
        `on_load_checkpoint` hook takes care of calling `setup`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__()

        # set-up audio IO
        assert (
            num_channels == 1
        ), "Only mono audio is supported for now (num_channels = 1)"
        self.hparams.sample_rate = sample_rate
        self.hparams.num_channels = num_channels
        self.audio = Audio(sample_rate=self.hparams.sample_rate, mono=True)

        # set task attribute when available (i.e. at training time)
        # and also tell the task what kind of audio is expected from
        # the model
        if task is not None:
            self.task = task
            self.task.audio = self.audio

    @cached_property
    def is_multi_task(self) -> bool:
        if hasattr(self, "task"):
            return self.task.is_multi_task
        return len(self.hparams.task_specifications) > 1

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    #  used by Tensorboard logger to log model graph
    @cached_property
    def example_input_array(self) -> torch.Tensor:
        return self.task.example_input_array

    def helper_introspect(
        self,
        specifications: TaskSpecification,
        task: str = None,
    ) -> ModelIntrospection:
        """Helper function for model introspection

        Parameters
        ----------
        specifications : TaskSpecification
        task : str, optional
            Task name.

        Returns
        -------
        introspection : ModelIntrospection
            Model introspection.
        """
        example_input_array = self.task.example_input_array
        batch_size, num_channels, num_samples = example_input_array.shape
        example_input_array = torch.randn(
            (1, num_channels, num_samples),
            dtype=example_input_array.dtype,
            layout=example_input_array.layout,
            device=example_input_array.device,
            requires_grad=False,
        )

        # dichotomic search of "min_num_samples"
        lower, upper, min_num_samples = 1, num_samples, None
        while True:
            num_samples = (lower + upper) // 2
            try:
                with torch.no_grad():
                    frames = self(example_input_array[:, :, :num_samples])
                if task is not None:
                    frames = frames[task]
            except Exception:
                lower = num_samples
            else:
                min_num_samples = num_samples
                if specifications.scale == Scale.FRAME:
                    _, min_num_frames, dimension = frames.shape
                elif specifications.scale == Scale.CHUNK:
                    min_num_frames, dimension = frames.shape
                else:
                    # should never happen
                    pass
                upper = num_samples

            if lower + 1 == upper:
                break

        # if "min_num_samples" is still None at this point, it means that
        # the forward pass always failed and raised an exception. most likely,
        # it means that there is a problem with the model definition.
        # we try again without catching the exception to help the end user debug
        # their model
        if min_num_samples is None:
            frames = self(example_input_array)

        # corner case for chunk-scale tasks
        if specifications.scale == Scale.CHUNK:
            return ModelIntrospection(
                min_num_samples=min_num_samples,
                min_num_frames=1,
                inc_num_samples=0,
                inc_num_frames=0,
                dimension=dimension,
            )

        # search reasonable upper bound for "inc_num_samples"
        while True:
            num_samples = 2 * min_num_samples
            example_input_array = torch.randn(
                (1, num_channels, num_samples),
                dtype=example_input_array.dtype,
                layout=example_input_array.layout,
                device=example_input_array.device,
                requires_grad=False,
            )
            with torch.no_grad():
                frames = self(example_input_array)
            if task is not None:
                frames = frames[task]
            num_frames = frames.shape[1]
            if num_frames > min_num_frames:
                break

        # dichotomic search of "inc_num_samples"
        lower, upper = min_num_samples, num_samples
        while True:
            num_samples = (lower + upper) // 2
            example_input_array = torch.randn(
                (1, num_channels, num_samples),
                dtype=example_input_array.dtype,
                layout=example_input_array.layout,
                device=example_input_array.device,
                requires_grad=False,
            )
            with torch.no_grad():
                frames = self(example_input_array)
            if task is not None:
                frames = frames[task]
            num_frames = frames.shape[1]
            if num_frames > min_num_frames:
                inc_num_frames = num_frames - min_num_frames
                inc_num_samples = num_samples - min_num_samples
                upper = num_samples
            else:
                lower = num_samples

            if lower + 1 == upper:
                break

        return ModelIntrospection(
            min_num_samples=min_num_samples,
            min_num_frames=min_num_frames,
            inc_num_samples=inc_num_samples,
            inc_num_frames=inc_num_frames,
            dimension=dimension,
        )

    def introspect(self) -> Union[ModelIntrospection, Dict[Text, ModelIntrospection]]:
        """Perform model introspection

        Returns
        -------
        introspection: ModelIntrospection or {str: ModelIntrospection} dict
            Model introspection or {task_name: introspection} dictionary for
            multi-task models.
        """

        task_specifications = self.hparams.task_specifications

        if self.is_multi_task:
            return {
                name: self.helper_introspect(specs, task=name)
                for name, specs in task_specifications.items()
            }

        return self.helper_introspect(task_specifications)

    def setup(self, stage=None):

        if stage == "fit":
            #  keep track of task specifications
            self.hparams.task_specifications = self.task.specifications

        # add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        self.build()

        if stage == "fit":
            # model introspection
            self.hparams.model_introspection = self.introspect()

            # TODO: raises an error in case of multiple tasks with different introspections

            # let task know about model introspection
            # so that its dataloader knows how to generate targets
            self.task.model_introspection = self.hparams.model_introspection

            # this is needed to support pytorch-lightning auto_lr_find feature
            # as it expects to find a "learning_rate" entry in model.hparams
            self.hparams.learning_rate = self.task.learning_rate

    def on_save_checkpoint(self, checkpoint):

        #  put everything pyannote.audio-specific under pyannote.audio
        #  to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "torch": torch.__version__,
                "pyannote.audio": __version__,
            },
            "model": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
        }

    @staticmethod
    def check_version(library: Text, theirs: Text, mine: Text):
        theirs = VersionInfo.parse(theirs)
        mine = VersionInfo.parse(mine)
        if theirs.major != mine.major:
            warnings.warn(
                f"Model was trained with {library} {theirs}, yours is {mine}. "
                f"Bad things will probably happen unless you update {library} to {theirs.major}.x."
            )
        if theirs.minor > mine.minor:
            warnings.warn(
                f"Model was trained with {library} {theirs}, yours is {mine}. "
                f"This should be OK but you might want to update {library}."
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):

        self.check_version(
            "pyannote.audio",
            checkpoint["pyannote.audio"]["versions"]["pyannote.audio"],
            __version__,
        )

        self.check_version(
            "torch",
            checkpoint["pyannote.audio"]["versions"]["torch"],
            torch.__version__,
        )
        self.check_version(
            "pytorch-lightning", checkpoint["pytorch-lightning_version"], pl.__version__
        )

        self.hparams.task_specifications = checkpoint["hyper_parameters"][
            "task_specifications"
        ]

        self.hparams.model_introspection = checkpoint["hyper_parameters"][
            "model_introspection"
        ]

        # now that setup()-defined hyper-parameters are available,
        # we can actually setup() the model.
        self.setup()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    def helper_default_activation(self, specifications: TaskSpecification) -> nn.Module:
        """Helper function for default_activation

        Parameters
        ----------
        specifications: TaskSpecification
            Task specification.

        Returns
        -------
        activation : nn.Module
            Default activation function.
        """

        if specifications.problem == Problem.BINARY_CLASSIFICATION:
            return nn.Sigmoid()

        elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return nn.LogSoftmax(dim=-1)

        elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return nn.Sigmoid()

        else:
            msg = "TODO: implement default activation for other types of problems"
            raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> Union[nn.Module, Dict[str, nn.Module]]:
        """Guess default activation function according to task specification

            * log-softmax for regular classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : nn.Module or {str: nn.Module}
            Activation or {task_name: activation} dictionary for multi-task models.
        """

        task_specifications = self.hparams.task_specifications

        if self.is_multi_task:
            return {
                name: self.helper_default_activation(specs)
                for name, specs in task_specifications.items()
            }

        return self.helper_default_activation(task_specifications)

    # training step logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def training_step(self, batch, batch_idx):
        return self.task.training_step(self, batch, batch_idx)

    # validation step logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def validation_step(self, batch, batch_idx):
        return self.task.validation_step(self, batch, batch_idx)

    # optimizer is delegated to the task for the same reason as above
    def configure_optimizers(self):
        return self.task.configure_optimizers(self)

    def _helper_up_to(
        self, module_name: Text, requires_grad: bool = False
    ) -> List[Text]:
        """Helper function for freeze_up_to and unfreeze_up_to"""

        tokens = module_name.split(".")
        updated_modules = list()

        for name, module in self.summarize("full").named_modules:
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
        return self._helper_up_to(module_name, requires_grad=False)

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
        return self._helper_up_to(module_name, requires_grad=True)

    def _helper_by_name(
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

        for name, module in self.summarize("full").named_modules:

            if name not in modules:
                continue

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad

            # keep track of updated modules
            updated_modules.append(name)

        missing = list(set(modules) - set(updated_modules))
        if missing:
            raise ValueError(f"Could not find the following modules: {missing}.")

        return updated_modules

    def freeze_by_name(
        self, modules: Union[Text, List[Text]], recurse: bool = True
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

        return self._helper_by_name(
            modules,
            recurse=recurse,
            requires_grad=False,
        )

    def unfreeze_by_name(
        self, modules: Union[List[Text], Text], recurse: bool = True
    ) -> List[Text]:
        """Unfreeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to unfreeze

        Returns
        -------
        unfrozen_modules: list of str
            Names of unfrozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self._helper_by_name(modules, recurse=recurse, requires_grad=True)


def load_from_checkpoint(checkpoint_path: str, map_location=None) -> Model:
    """Load model from checkpoint

    Parameters
    ----------
    checkpoint_path: str
        Path to checkpoint. This can also be a URL.

    Returns
    -------
    model : Model
        Model
    """

    # obtain model class from the checkpoint
    checkpoint = pl_load(checkpoint_path, map_location=map_location)

    module_name: str = checkpoint["pyannote.audio"]["model"]["module"]
    module = import_module(module_name)

    class_name: str = checkpoint["pyannote.audio"]["model"]["class"]
    Klass: Model = getattr(module, class_name)

    return Klass.load_from_checkpoint(checkpoint_path)
