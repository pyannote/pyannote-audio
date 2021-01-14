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


from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Optional, Text

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.database import Protocol


# Type of machine learning problem
class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4
    # any other we could think of?


# A task takes an audio chunk as input and returns
# either a temporal sequence of predictions
# or just one prediction for the whole audio chunk
class Scale(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk


@dataclass
class Specifications:
    problem: Problem
    scale: Scale

    # chunk duration in seconds.
    # use None for variable-length chunks
    duration: Optional[float] = None

    # (for classification tasks only) list of classes
    classes: Optional[List[Text]] = None

    # whether classes are permutation-invariant (e.g. diarization)
    permutation_invariant: bool = False

    def __len__(self):
        # makes it possible to do something like:
        # multi_task = len(task_specifications) > 1
        # because multi-task specifications are stored as {task_name: specifications} dict
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


class TrainDataset(IterableDataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.train__iter__()

    def __len__(self):
        return self.task.train__len__()


class ValDataset(Dataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __getitem__(self, idx):
        return self.task.val__getitem__(idx)

    def __len__(self):
        return self.task.val__len__()


class Task(pl.LightningDataModule):
    """Base task class

    A task is the combination of a "problem" and a "dataset".
    For example, here are a few tasks:
    - voice activity detection on the AMI corpus
    - speaker embedding on the VoxCeleb corpus
    - end-to-end speaker diarization on the VoxConverse corpus

    A task is expected to be solved by a "model" that takes an
    audio chunk as input and returns the solution. Hence, the
    task is in charge of generating (input, expected_output)
    samples used for training the model.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration in seconds. Defaults to two seconds (2.).
    min_duration : float, optional
        Sample training chunks duration uniformely between `min_duration`
        and `duration`. Defaults to `duration` (i.e. fixed length chunks).
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.

    Attributes
    ----------
    specifications : Specifications or dict of Specifications
        Task specifications (available after `Task.setup` has been called.)
        For multi-task learning, this should be a dictionary where keys are
        task names and values are corresponding Specifications instances.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        min_duration: float = None,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
        augmentation: BaseWaveformTransform = None,
    ):
        super().__init__()

        # dataset
        self.protocol = protocol

        # TODO: check that protocol files come with required fields: audio, annotation, annotated, uri, what else?

        # batching
        self.duration = duration
        self.min_duration = duration if min_duration is None else min_duration
        self.batch_size = batch_size

        # multi-processing
        if (
            num_workers > 0
            and sys.platform == "darwin"
            and sys.version_info[0] >= 3
            and sys.version_info[1] >= 8
        ):
            warnings.warn(
                "num_workers > 0 is not supported with macOS and Python 3.8+: "
                "setting num_workers = 0."
            )
            num_workers = 0

        self.num_workers = num_workers

        self.pin_memory = pin_memory

        if optimizer is None:
            optimizer = Adam
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.augmentation = augmentation

    def prepare_data(self):
        """Use this to download and prepare data

        This is where we might end up downloading datasets
        and transform them so that they are ready to be used
        with pyannote.database. but for now, the API assume
        that we directly provide a pyannote.database.Protocol.

        Notes
        -----
        Called only once.
        """
        pass

    def setup(self, stage=None):
        """Called at the beginning of fit and test just before Model.setup()

        Parameters
        ----------
        stage : "fit" or "test"
            Whether model is being trained ("fit") or used for inference ("test").

        Notes
        -----
        This hook is called on every process when using DDP.

        If `specifications` attribute has not been set in `__init__`,
        `setup` is your last chance to set it.
        """
        pass

    def setup_loss_func(self):
        pass

    def setup_validation_metric(self):
        pass

    @property
    def is_multi_task(self) -> bool:
        """"Check whether multiple tasks are addressed at once"""
        return len(self.specifications) > 1

    def train__iter__(self):
        # will become train_dataset.__iter__ method
        msg = f"Missing '{self.__class__.__name__}.train__iter__' method."
        raise NotImplementedError(msg)

    def train__len__(self):
        # will become train_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.train__len__' method."
        raise NotImplementedError(msg)

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)
        if self.augmentation is not None:
            collated_batch["X"] = self.augmentation(
                collated_batch["X"], sample_rate=self.model.sample_rate
            )
        return collated_batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TrainDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def default_loss(self, specifications: Specifications, y, y_pred) -> torch.Tensor:
        """Guess and compute default loss according to task specification"""

        if specifications.problem == Problem.BINARY_CLASSIFICATION:
            loss = F.binary_cross_entropy(y_pred.squeeze(dim=-1), y.float())

        elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            loss = F.nll_loss(y_pred.view(-1, len(specifications.classes)), y.view(-1))

        elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            loss = F.binary_cross_entropy(y_pred, y.float())

        else:
            msg = "TODO: implement for other types of problems"
            raise NotImplementedError(msg)

        return loss

    # default training_step provided for convenience
    # can obviously be overriden for each task
    def training_step(self, batch, batch_idx: int):
        """Default training_step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        In case of multi-tasking, it will default to summing loss of each task.

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "loss_{task_name}" keys for multi-task models.
        """

        X, y = batch["X"], batch["y"]
        y_pred = self.model(X)

        if self.is_multi_task:
            loss = dict()
            for task_name, specifications in self.specifications.items():
                loss[task_name] = self.default_loss(
                    specifications, y[task_name], y_pred[task_name]
                )
                self.model.log(
                    f"{task_name}@train_loss",
                    loss[task_name],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=False,
                )

            loss["loss"] = sum(loss.values())
            self.model.log(
                f"{self.ACRONYM}@train_loss",
                loss["loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

        loss = self.default_loss(self.specifications, y, y_pred)
        self.model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def val__getitem__(self, idx):
        # will become val_dataset.__getitem__ method
        msg = f"Missing '{self.__class__.__name__}.val__getitem__' method."
        raise NotImplementedError(msg)

    def val__len__(self):
        # will become val_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.val__len__' method."
        raise NotImplementedError(msg)

    def val_dataloader(self) -> Optional[DataLoader]:
        val_callback = self.val_callback()
        if val_callback is None:
            return DataLoader(
                ValDataset(self),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
        else:
            return None

    # default validation_step provided for convenience
    # can obviously be overriden for each task
    def validation_step(self, batch, batch_idx: int):
        """Guess default validation_step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        In case of multi-tasking, it will default to summing loss of each task.

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "{task_name}" keys for multi-task models.
        """

        X, y = batch["X"], batch["y"]
        y_pred = self.model(X)

        if self.is_multi_task:
            loss = dict()
            for task_name, specifications in self.specifications.items():
                loss[task_name] = self.default_loss(
                    specifications, y[task_name], y_pred[task_name]
                )
                self.model.log(f"{task_name}@val_loss", loss[task_name])

            loss["loss"] = sum(loss.values())
            self.model.log(
                f"{self.ACRONYM}@val_loss",
                loss["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

        loss = self.default_loss(self.specifications, y, y_pred)
        self.model.log(
            f"{self.ACRONYM}@val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        pass

    def val_callback(self):
        return None

    # default configure_optimizers provided for convenience
    # can obviously be overriden for each task
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.learning_rate)

    @property
    def val_monitor(self):
        """Quantity (and direction) to monitor

        Useful for model checkpointing or early stopping.

        Returns
        -------
        monitor : str
            Name of quantity to monitor.
        mode : {'min', 'max}
            Minimize

        See also
        --------
        pytorch_lightning.callbacks.ModelCheckpoint
        pytorch_lightning.callbacks.EarlyStopping
        """
        return f"{self.ACRONYM}@val_loss", "min"
