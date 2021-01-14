# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union
from urllib.parse import urlparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from huggingface_hub import cached_download, hf_hub_url
from pytorch_lightning.utilities.cloud_io import load as pl_load
from semver import VersionInfo

from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Scale, Specifications, Task

CACHE_DIR = os.getenv(
    "PYANNOTE_CACHE",
    os.path.expanduser("~/.cache/torch/pyannote"),
)
HF_PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"


@dataclass
class Introspection:

    # TODO: make it a regular class
    # TODO: add from_model class method that will replace Model.introspection property

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
        # multi_task = len(introspection) > 1
        # because multi-task introspections are stored as {task_name: introspection} dict
        return 1

    def __getitem__(self, key):
        if key is not None:
            raise KeyError
        return self

    def items(self):
        yield None, self

    def keys(self):
        yield None

    def __iter__(self):
        yield None


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
        self.audio = Audio(
            sample_rate=self.hparams.sample_rate, mono=self.hparams.num_channels == 1
        )

    @property
    def datamodule(self):
        return self.task

    @property
    def example_input_array(self) -> torch.Tensor:
        batch_size = 3 if self.task is None else self.task.batch_size
        duration = self.specifications.duration
        return torch.randn(
            (
                batch_size,
                self.hparams.num_channels,
                int(self.hparams.sample_rate * duration),
            )
        )

        return self.task.example_input_array

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        self._task = task
        del self.introspection
        del self.specifications

    @property
    def specifications(self):
        if self.task is not None:
            return self.task.specifications
        return self._specifications

    @specifications.setter
    def specifications(self, specifications):
        self._specifications = specifications

    @specifications.deleter
    def specifications(self):
        if hasattr(self, "_specifications"):
            del self._specifications

    @property
    def is_multi_task(self):
        return len(self.specifications) > 1

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    def helper_introspection(
        self,
        specifications: Specifications,
        task: str = None,
    ) -> Introspection:
        """Helper function for model introspection

        Parameters
        ----------
        specifications : Specifications
            Task specifications.
        task : str, optional
            Task name.

        Returns
        -------
        introspection : Introspection
            Model introspection.
        """

        example_input_array = self.example_input_array
        batch_size, num_channels, num_samples = example_input_array.shape
        example_input_array = torch.randn(
            (1, num_channels, num_samples),
            dtype=example_input_array.dtype,
            layout=example_input_array.layout,
            device=example_input_array.device,
            requires_grad=False,
        ).to(self.device)

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
            return Introspection(
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

        return Introspection(
            min_num_samples=min_num_samples,
            min_num_frames=min_num_frames,
            inc_num_samples=inc_num_samples,
            inc_num_frames=inc_num_frames,
            dimension=dimension,
        )

    @property
    def introspection(self) -> Union[Introspection, Dict[Text, Introspection]]:
        """Introspection

        Returns
        -------
        introspection: Introspection or {str: Introspection} dict
            Model introspection or {task_name: introspection} dictionary for
            multi-task models.
        """

        if not hasattr(self, "_introspection"):

            if self.is_multi_task:
                self._introspection = {
                    name: self.helper_introspection(specs, task=name)
                    for name, specs in self.specifications.items()
                }
                # TODO: raises an error in case of multiple tasks with different introspections

            self._introspection = self.helper_introspection(self.specifications)

        return self._introspection

    @introspection.setter
    def introspection(self, introspection):
        self._introspection = introspection

    @introspection.deleter
    def introspection(self):
        if hasattr(self, "_introspection"):
            del self._introspection

    def setup(self, stage=None):

        self.build()

        if stage == "fit":
            self.task.model = self
            self.task.setup_loss_func()
            self.task.setup_validation_metric()

    def on_save_checkpoint(self, checkpoint):

        #  put everything pyannote.audio-specific under pyannote.audio
        #  to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "torch": torch.__version__,
                "pyannote.audio": __version__,
            },
            "architecture": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
            "introspection": self.introspection,
            "specifications": self.specifications,
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

        self.introspection = checkpoint["pyannote.audio"]["introspection"]
        self.specifications = checkpoint["pyannote.audio"]["specifications"]

        self.build()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    def helper_default_activation(self, specifications: Specifications) -> nn.Module:
        """Helper function for default_activation

        Parameters
        ----------
        specifications: Specifications
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

            * sigmoid for binary classification
            * log-softmax for regular multi-class classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : nn.Module or {str: nn.Module}
            Activation or {task_name: activation} dictionary for multi-task models.
        """

        if self.is_multi_task:
            return nn.ModuleDict(
                {
                    name: self.helper_default_activation(specs)
                    for name, specs in self.specifications.items()
                }
            )

        return self.helper_default_activation(self.specifications)

    # training step logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def training_step(self, batch, batch_idx):
        return self.task.training_step(batch, batch_idx)

    # validation logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def validation_step(self, batch, batch_idx):
        return self.task.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.task.validation_epoch_end(outputs)

    # optimizer is delegated to the task for the same reason as above
    def configure_optimizers(self):
        return self.task.configure_optimizers()

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


def load_from_checkpoint(
    checkpoint_path: Union[Path, Text],
    map_location=None,
    hparams_file: Union[Path, Text] = None,
    strict: bool = True,
    **kwargs,
) -> Model:
    """Load model from checkpoint

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to checkpoint, or a remote URL, or a model identifier from
        the huggingface.co model hub.
    map_location: optional
        If your checkpoint saved a GPU model and you now load on CPUs
        or a different number of GPUs, use this to map to the new setup.
        The behaviour is the same as in torch.load().
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
        Whether to strictly enforce that the keys in checkpoint_path match
        the keys returned by this module’s state dict. Defaults to True.
    kwargs: optional
        Any extra keyword args needed to init the model.
        Can also be used to override saved hyperparameter values.

    Returns
    -------
    model : Model
        Model
    """

    # pytorch-lightning expects str, not Path.
    checkpoint_path = str(checkpoint_path)
    if hparams_file is not None:
        hparams_file = str(hparams_file)

    # resolve the checkpoint_path to
    # something that pl will handle
    if os.path.isfile(checkpoint_path):
        path_for_pl = checkpoint_path
    elif urlparse(checkpoint_path).scheme in ("http", "https"):
        path_for_pl = checkpoint_path
    else:
        # Finally, let's try to find it on Hugging Face model hub
        # e.g. julien-c/voice-activity-detection is a valid model id
        # and  julien-c/voice-activity-detection@main supports specifying a commit/branch/tag.
        if "@" in checkpoint_path:
            model_id = checkpoint_path.split("@")[0]
            revision = checkpoint_path.split("@")[1]
        else:
            model_id = checkpoint_path
            revision = None
        url = hf_hub_url(
            model_id=model_id, filename=HF_PYTORCH_WEIGHTS_NAME, revision=revision
        )
        path_for_pl = cached_download(
            url=url,
            library_name="pyannote",
            library_version=__version__,
            cache_dir=CACHE_DIR,
        )

    # obtain model class from the checkpoint
    checkpoint = pl_load(path_for_pl, map_location=map_location)

    module_name: str = checkpoint["pyannote.audio"]["architecture"]["module"]
    module = import_module(module_name)

    class_name: str = checkpoint["pyannote.audio"]["architecture"]["class"]
    Klass: Model = getattr(module, class_name)

    return Klass.load_from_checkpoint(
        path_for_pl,
        map_location=map_location,
        hparams_file=hparams_file,
        strict=strict,
        **kwargs,
    )
