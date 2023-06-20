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

import multiprocessing
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from numbers import Number
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import scipy.special
import torch
from pyannote.database import Protocol
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_audiomentations import Identity
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, MetricCollection

from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.protocol import check_protocol


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


class UnknownSpecificationsError(Exception):
    pass


@dataclass
class Specifications:
    problem: Problem
    resolution: Resolution

    # (maximum) chunk duration in seconds
    duration: float

    # (for variable-duration tasks only) minimum chunk duration in seconds
    min_duration: Optional[float] = None

    # use that many seconds on the left- and rightmost parts of each chunk
    # to warm up the model. This is mostly useful for segmentation tasks.
    # While the model does process those left- and right-most parts, only
    # the remaining central part of each chunk is used for computing the
    # loss during training, and for aggregating scores during inference.
    # Defaults to 0. (i.e. no warm-up).
    warm_up: Optional[Tuple[float, float]] = (0.0, 0.0)

    # (for classification tasks only) list of classes
    classes: Optional[List[Text]] = None

    # (for powerset only) max number of simultaneous classes
    # (n choose k with k <= powerset_max_classes)
    powerset_max_classes: Optional[int] = None

    # whether classes are permutation-invariant (e.g. diarization)
    permutation_invariant: bool = False

    @cached_property
    def powerset(self) -> bool:
        if self.powerset_max_classes is None:
            return False

        if self.problem != Problem.MONO_LABEL_CLASSIFICATION:
            raise ValueError(
                "`powerset_max_classes` only makes sense with multi-class classification problems."
            )

        return True

    @cached_property
    def num_powerset_classes(self) -> int:
        # compute number of subsets of size at most "powerset_max_classes"
        # e.g. with len(classes) = 3 and powerset_max_classes = 2:
        # {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
        return int(
            sum(
                scipy.special.binom(len(self.classes), i)
                for i in range(0, self.powerset_max_classes + 1)
            )
        )

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


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
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to value returned by `default_metric` method.

    Attributes
    ----------
    specifications : Specifications or tuple of Specifications
        Task specifications (available after `Task.setup` has been called.)
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
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):
        super().__init__()

        # dataset
        self.protocol, checks = check_protocol(protocol)
        self.has_validation = checks["has_validation"]
        self.has_scope = checks["has_scope"]
        self.has_classes = checks["has_classes"]

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
        self.augmentation = augmentation or Identity(output_type="dict")
        self._metric = metric

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

    @property
    def specifications(self) -> Union[Specifications, Tuple[Specifications]]:
        # setup metadata on-demand the first time specifications are requested and missing
        if not hasattr(self, "_specifications"):
            self.setup_metadata()
        return self._specifications

    @specifications.setter
    def specifications(
        self, specifications: Union[Specifications, Tuple[Specifications]]
    ):
        self._specifications = specifications

    @property
    def has_setup_metadata(self):
        return getattr(self, "_has_setup_metadata", False)

    @has_setup_metadata.setter
    def has_setup_metadata(self, value: bool):
        self._has_setup_metadata = value

    def setup_metadata(self):
        """Called at the beginning of training at the very beginning of Model.setup(stage="fit")

        Notes
        -----
        This hook is called on every process when using DDP.

        If `specifications` attribute has not been set in `__init__`,
        `setup` is your last chance to set it.
        """

        if not self.has_setup_metadata:
            self.setup()
            self.has_setup_metadata = True

    def setup_loss_func(self):
        pass

    def train__iter__(self):
        # will become train_dataset.__iter__ method
        msg = f"Missing '{self.__class__.__name__}.train__iter__' method."
        raise NotImplementedError(msg)

    def train__len__(self):
        # will become train_dataset.__len__ method
        msg = f"Missing '{self.__class__.__name__}.train__len__' method."
        raise NotImplementedError(msg)

    def collate_fn(self, batch, stage="train"):
        msg = f"Missing '{self.__class__.__name__}.collate_fn' method."
        raise NotImplementedError(msg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TrainDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=partial(self.collate_fn, stage="train"),
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

        elif specifications.problem in [Problem.MONO_LABEL_CLASSIFICATION]:
            return nll_loss(prediction, target, weight=weight)

        else:
            msg = "TODO: implement for other types of problems"
            raise NotImplementedError(msg)

    def common_step(self, batch, batch_idx: int, stage: Literal["train", "val"]):
        """Default training or validation step according to task specification

            * binary cross-entropy loss for binary or multi-label classification
            * negative log-likelihood loss for regular classification

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
            {"loss": loss}
        """

        if isinstance(self.specifications, tuple):
            raise NotImplementedError(
                "Default training/validation step is not implemented for multi-task."
            )

        # forward pass
        y_pred = self.model(batch["X"])

        batch_size, num_frames, _ = y_pred.shape
        # (batch_size, num_frames, num_classes)

        # target
        y = batch["y"]

        # frames weight
        weight_key = getattr(self, "weight", None) if stage == "train" else None
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        # compute loss
        loss = self.default_loss(self.specifications, y, y_pred, weight=weight)

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            f"loss/{stage}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
                collate_fn=partial(self.collate_fn, stage="val"),
            )
        else:
            return None

    # default validation_step provided for convenience
    # can obviously be overriden for each task
    def validation_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "val")

    def default_metric(self) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Default validation metric"""
        msg = f"Missing '{self.__class__.__name__}.default_metric' method."
        raise NotImplementedError(msg)

    @cached_property
    def metric(self) -> MetricCollection:
        if self._metric is None:
            self._metric = self.default_metric()

        return MetricCollection(self._metric)

    def setup_validation_metric(self):
        metric = self.metric
        if metric is not None:
            self.model.validation_metric = metric
            self.model.validation_metric.to(self.model.device)

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

        name, metric = next(iter(self.metric.items()))
        return name, "max" if metric.higher_is_better else "min"
