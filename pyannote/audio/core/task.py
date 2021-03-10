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


from __future__ import annotations

import multiprocessing
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import List, Optional, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing_extensions import Literal

from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.protocol import check_protocol
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
class Resolution(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk


@dataclass
class Specifications:
    problem: Problem
    resolution: Resolution

    # chunk duration in seconds.
    # use None for variable-length chunks
    duration: Optional[float] = None

    # use that many seconds on the left- and rightmost parts of each chunk
    # to warm up the model. This is mostly useful for segmentation tasks.
    # While the model does process those left- and right-most parts, only
    # the remaining central part of each chunk is used for computing the
    # loss during training, and for aggregating scores during inference.
    # Defaults to 0. (i.e. no warm-up).
    warm_up: Optional[Tuple[float, float]] = (0.0, 0.0)

    # (for classification tasks only) list of classes
    classes: Optional[List[Text]] = None

    # whether classes are permutation-invariant (e.g. diarization)
    permutation_invariant: bool = False

    def __len__(self):
        # makes it possible to do something like:
        # multi_task = len(specifications) > 1
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
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. This is mostly useful for segmentation tasks.
        While the model does process those left- and right-most parts, only
        the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
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
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
    ):
        super().__init__()

        # dataset
        self.protocol, self.has_validation = check_protocol(protocol)

        # batching
        self.duration = duration
        self.min_duration = duration if min_duration is None else min_duration
        self.batch_size = batch_size

        # training
        if isinstance(warm_up, Number):
            warm_up = (warm_up, warm_up)
        self.warm_up = warm_up

        # multi-processing
        if num_workers is None:
            num_workers = multiprocessing.cpu_count() // 2

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
                collated_batch["X"], sample_rate=self.model.hparams.sample_rate
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

    def default_loss(
        self, specifications: Specifications, target, prediction, weight=None
    ) -> torch.Tensor:
        """Guess and compute default loss according to task specification

        Parameters
        ----------
        specifications : Specifications
            Task specifications
        target : torch.Tensor
            * (batch_size, num_frames) for binary classification
            * (batch_size, num_frames) for multi-class classification
            * (batch_size, num_frames, num_classes) for multi-label classification
        prediction : torch.Tensor
            (batch_size, num_frames, num_classes)
        weight : torch.Tensor, optional
            (batch_size, num_frames, 1)

        Returns
        -------
        loss : torch.Tensor
            Binary cross-entropy loss in case of binary and multi-label classification,
            Negative log-likelihood loss in case of multi-class classification.

        """

        if specifications.problem in [
            Problem.BINARY_CLASSIFICATION,
            Problem.MULTI_LABEL_CLASSIFICATION,
        ]:
            return binary_cross_entropy(prediction, target, weight=weight)

        elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return nll_loss(prediction, target, weight=weight)

        else:
            msg = "TODO: implement for other types of problems"
            raise NotImplementedError(msg)

    def common_step(self, batch, batch_idx: int, stage: Literal["train", "val"]):
        """Default training or validation step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

        In case of multi-tasking, it will default to summing loss of each task.

        If "weight" attribute exists, batch[self.weight] is also passed to the loss function
        during training (but has no effect in validation).

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        stage : {"train", "val"}
            "train" for training step, "val" for validation step

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "loss_{task_name}" keys for multi-task models.
        """

        # forward pass
        y_pred = self.model(batch["X"])

        if self.is_multi_task:
            _, any_y_pred = next(iter(y_pred.items()))
            batch_size, num_frames, _ = any_y_pred.shape
        else:
            batch_size, num_frames, _ = y_pred.shape
            # (batch_size, num_frames, num_classes)

        # target
        y = batch["y"]

        # frames weight
        weight_key = getattr(self, "weight", None) if stage == "train" else None
        weight = batch.get(weight_key, torch.ones(batch_size, num_frames, 1))
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        # compute multi-task loss as the sum of loss of each task
        if self.is_multi_task:
            loss = dict()
            for task_name, specifications in self.specifications.items():
                loss[task_name] = self.default_loss(
                    specifications, y[task_name], y_pred[task_name], weight=weight
                )
                self.model.log(
                    f"{task_name}@{stage}_loss",
                    loss[task_name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=False,
                )

            loss["loss"] = sum(loss.values())
            self.model.log(
                f"{self.ACRONYM}@{stage}_loss",
                loss["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

        # compute mono-task loss
        loss = self.default_loss(self.specifications, y, y_pred, weight=weight)
        self.model.log(
            f"{self.ACRONYM}@{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    # default training_step provided for convenience
    # can obviously be overriden for each task
    def training_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "train")

    def val__getitem__(self, idx):
        # will become val_dataset.__getitem__ method
        msg = f"Missing '{self.__class__.__name__}.val__getitem__' method."
        raise NotImplementedError(msg)

    def val__len__(self):
        # will become val_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.val__len__' method."
        raise NotImplementedError(msg)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.has_validation:
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
        return self.common_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        pass

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
        if self.has_validation:
            return f"{self.ACRONYM}@val_loss", "min"
        else:
            return None, "min"
