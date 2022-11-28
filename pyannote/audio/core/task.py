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

from functools import partial
import itertools
import math
import scipy.special

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property

import multiprocessing
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_audiomentations import Identity
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, MetricCollection
from typing_extensions import Literal

from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.protocol import check_protocol


# Type of machine learning problem
class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4
    POWERSET = 5
    # any other we could think of?

    @staticmethod
    def compute_powerset_conversion_dict(
        max_num_speakers: int, max_simult_speakers: int
    ) -> Dict[int, tuple]:
        """Returns a dict that maps all powerset classes to tuples of active multilabel speaker number.

        Parameters
        ----------
        max_num_speakers : int
            Number of distinct speaker identities
        max_simult_speakers : int
            Maximum number of speakers that can be active simultaneously

        Returns
        -------
        Dict[int,tuple]
            The mapping 'class -> tuple of active speakers'. The speaker id is in [0,max_num_speakers-1]
        """
        powerset_to_multi = {0: ()}  # id==0 : "no speaker" class
        speakers = [i for i in range(max_num_speakers)]

        id = 1  # begin at 1, id==0 is "no speaker"
        for simult in range(1, max_simult_speakers + 1):
            # all combinations of simult speakers
            for c in itertools.combinations(speakers, simult):
                powerset_to_multi[id] = c
                id += 1  # one combination = one id
        return powerset_to_multi

    @staticmethod
    def get_powerset_class_count(
        max_num_speakers: int, max_simult_speakers: int
    ) -> int:
        """For the given parameters, get how many classes the powerset encoding contains.

        Parameters
        ----------
        max_num_speakers : int
            Number of distinct speaker identities
        max_simult_speakers : int
            Maximum number of speakers that can be active simultaneously

        Returns
        -------
        int
            Number of classes in the powerset encoding
        """

        result = 0  # account for "no speaker" class
        for i in range(0, max_simult_speakers + 1):
            result += int(scipy.special.binom(max_num_speakers, i))
            # result += math.comb(max_num_speakers, i)  # python >=3.8 only
        return result

    @staticmethod
    def build_powerset_to_multi_conversion_tensor(
        max_num_speakers: int, max_simult_speakers: int, device: torch.device = None
    ) -> torch.Tensor:
        """Builds a conversion tensor of size [num_classes_powerset, max_num_speakers].
        For each row (which corresponding to a powerset class), the active speakers in that row
        have their corresponding column set to 1.0, inactive speakers have theirs set to 0.0.

        For example, with 3 max simultaneous speakers, row [0., 0., 1.] indicates that the
        speaker 2 is active in that class.

        Parameters
        ----------
        max_num_speakers : int
            Number of distinct speaker identities
        max_simult_speakers : int
            Maximum number of speakers that can be active simultaneously
        device : torch.device, optional
            Device to build the conversion tensor on, by default None

        Returns
        -------
        torch.Tensor
            The [num_classes_powerset, max_num_speakers]-shaped powerset <-> multilabel conversion tensor
        """

        powerset_to_multi = __class__.compute_powerset_conversion_dict(
            max_num_speakers, max_simult_speakers
        )

        a = torch.zeros(len(powerset_to_multi), max_num_speakers, device=device).float()
        for id in powerset_to_multi:
            speakers = powerset_to_multi[id]
            if len(speakers) > 0:
                a[id][torch.tensor(speakers)] = 1.0
        return a

    @staticmethod
    def multilabel_to_powerset(
        t: torch.Tensor,
        max_num_speakers: int,
        max_simult_speakers: int,
        conversion_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """Takes as input a multilabel tensor and outputs its corresponding one-hot powerset tensor.

        Parameters
        ----------
        t : torch.Tensor
            (BATCH_SIZE,NUM_FRAMES,NUM_SPEAKERS) tensor
        max_num_speakers : int
            Maximum number of different speakers in a batch
        max_simult_speakers : int
            Maximum number of simultaneously active speakers in one frame
        conversion_tensor: torch.Tensor
            The tensor built with 'build_powerset_to_multi_conversion_tensor' (to avoid recomputing it each call)

        Returns
        -------
        torch.Tensor
            One hot (BATCH_SIZE,NUM_FRAMES,NUM_CLASSES_POWERSET) tensor
        """

        # if torch.max(torch.sum(t, dim=2).flatten()) > max_simult_speakers:
        #     print(f"Warning : more than {max_simult_speakers} simult speakers ! {torch.max(torch.sum(t, dim=2).flatten())}")
        if t.shape[-1] > max_num_speakers:
            print(
                "WARNING: input tensor has too many speakers. Blindly removing the last ones"
            )
            t = t[:, :, :max_num_speakers]
        else:
            t = torch.nn.functional.pad(t, [0, max_num_speakers - t.shape[-1]])

        if conversion_tensor is None:
            conversion_tensor = __class__.build_powerset_to_multi_conversion_tensor(
                max_num_speakers, max_simult_speakers, device=t.device
            )
        num_powerset_classes = conversion_tensor.shape[0]

        # multiply the tensor by the conversion tensor and take the argmax to find which class is active
        # in case multiple elts are equal, we rely on the argmax implementation where the first elt with
        # that value is taken.
        # (eg. after multiplying, if speaker 1 is active, both classes for spk 1 and spk 1+2+3 will be == 1
        # but we want to take the class spk 1, which is why classes are ordered in this way. Same problem
        # with 0 speakers active)
        multiplied = torch.matmul(t.float(), conversion_tensor.t())
        argmaxed = torch.argmax(multiplied, dim=-1)
        result = torch.nn.functional.one_hot(
            argmaxed.long(), num_classes=num_powerset_classes
        )

        return result

    @staticmethod
    def powerset_to_multilabel(
        ps_t: torch.Tensor,
        max_num_speakers: int,
        max_simult_speakers: int,
        conversion_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """Converts powerset encoding into multilabel tensor.
        Should probably only be used to convert one hot encodings into multi-hot encoding,
        other uses do not make sense.

        Parameters
        ----------
        ps_t : torch.Tensor
            (BATCH_SIZE,NUM_FRAMES,NUM_CLASSES_POWERSET) tensor (one-hot)
        max_num_speakers : int
            Maximum number of different speakers in a batch
        max_simult_speakers : int
            Maximum number of simultaneously active speakers in one frame
        conversion_tensor: torch.Tensor
            The tensor built with 'build_powerset_to_multi_conversion_tensor' (to avoid recomputing it each call)


        Returns
        -------
        torch.Tensor
            (BATCH_SIZE,NUM_FRAMES,MAX_NUM_SPEAKERS) tensor
        """

        # input: (B,F,Classes)
        # output: (B,F,max_num_speakers)
        num_batches, num_frames, num_classes = ps_t.shape

        if conversion_tensor is None:
            conversion_tensor = __class__.build_powerset_to_multi_conversion_tensor(
                max_num_speakers, max_simult_speakers, device=ps_t.device
            )

        result = torch.matmul(ps_t.float(), conversion_tensor)

        return result

    @staticmethod
    def get_powerset_permutation(
        permutation: torch.Tensor,
        max_speakers: int,
        max_simult_speakers: int,
        conv_dict: dict = None,
        inv_conv_dict: dict = None,
    ) -> List[int]:
        """Converts a multilabel permutation into a powerset permutation.

        Parameters
        ----------
        permutation : torch.Tensor
            The permutation, of shape (<=MAX_SPEAKERS)
        max_speakers : int
            Number of distinct speaker identities
        max_simult_speakers : int
            Maximum number of speakers that can be active simultaneously
        conv_dict : dict, optional
            The conversion dictionary built with 'compute_powerset_conversion_dict' (to avoid recomputing it each call), by default None
        inv_conv_dict : dict, optional
            The inverse mapping of conv_dict  (to avoid recomputing it each call), by default None

        Returns
        -------
        List[int]
            _description_
        """

        # In case the permutation only keeps some speakers, build a tensor padded_permutation
        # made of the permutation followed by the unused speakers
        arange_t, idx_counts = torch.cat(
            [torch.arange(0, max_speakers), permutation]
        ).unique(return_counts=True)
        padded_permutation = torch.cat([permutation, arange_t[idx_counts == 1]])

        # build the conversion dicts if necessary
        if conv_dict is None:
            conv_dict = __class__.compute_powerset_conversion_dict(
                max_speakers, max_simult_speakers
            )
        if inv_conv_dict is None:
            inv_conv_dict = {v: k for k, v in conv_dict.items()}

        perm_powerset = [
            0,
        ]
        speakers = [i for i in range(max_speakers)]
        for simult in range(1, max_simult_speakers + 1):
            # all combinations of simult speakers
            for c in itertools.combinations(speakers, simult):
                c_perm_t, _ = torch.sort(padded_permutation[torch.tensor(c)])
                c_perm = tuple(c_perm_t.tolist())
                perm_powerset.append(inv_conv_dict[c_perm])
        return perm_powerset


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
    # (for powerset only) max number of simultaneous active speakers (one speaker=one class in 'classes')
    max_simult_speakers: Optional[int] = None

    # whether classes are permutation-invariant (e.g. diarization)
    permutation_invariant: bool = False


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
    specifications : Specifications or dict of Specifications
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

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of training at the very beginning of Model.setup(stage="fit")

        Notes
        -----
        This hook is called on every process when using DDP.

        If `specifications` attribute has not been set in `__init__`,
        `setup` is your last chance to set it.
        """
        pass

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

    @cached_property
    def logging_prefix(self):

        prefix = f"{self.__class__.__name__}-"
        if hasattr(self.protocol, "name"):
            # "." has a special meaning for pytorch-lightning checkpointing
            # so we remove dots from protocol names
            name_without_dots = "".join(self.protocol.name.split("."))
            prefix += f"{name_without_dots}-"

        return prefix

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
            Problem.POWERSET,
        ]:
            return binary_cross_entropy(prediction, target, weight=weight)

        elif (
            specifications.problem == Problem.MONO_LABEL_CLASSIFICATION
            or specifications.problem == Problem.POWERSET
        ):
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
        self.model.log(
            f"{self.logging_prefix}{stage.capitalize()}Loss",
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
                collate_fn=partial(self.collate_fn, stage="val"),
            )
        else:
            return None

    # default validation_step provided for convenience
    # can obviously be overriden for each task
    def validation_step(self, batch, batch_idx: int):
        return self.common_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        pass

    def default_metric(self) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Default validation metric"""
        msg = f"Missing '{self.__class__.__name__}.default_metric' method."
        raise NotImplementedError(msg)

    @cached_property
    def metric(self) -> MetricCollection:
        if self._metric is None:
            self._metric = self.default_metric()

        return MetricCollection(self._metric, prefix=self.logging_prefix)

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
