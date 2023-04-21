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

from functools import cached_property
import math
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union
from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.database import Protocol
from pyannote.database.protocol import SegmentationProtocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import (
    CalibrationError,
    F1Score,
    Metric,
    Precision,
    Recall,
    MetricCollection,
)
from torchmetrics import F1Score, Metric, MetricCollection, Precision, Recall
from pytorch_lightning.loggers import TensorBoardLogger

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class Loggable:
    def __init__(
        self,
        name: str = "Histogram",
        update_in: Literal["train", "val"] = "val",
        log_on: Literal["step", "epoch"] = "epoch",
    ):
        self.name = name
        self.update_in = update_in
        self.compute_on = log_on

    # TODO: use kwargs ? also, pass task/model in calls
    def update(
        self,
        data: dict,
    ):
        raise NotImplementedError()

    def log(self, loggers: list, current_epoch: int, batch_idx: int):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class LoggableHistogram(Loggable):
    def __init__(
        self,
        bins: torch.Tensor,
        values_field: str = "y_pred",
        name: str = "Histogram",
        update_in: Literal["train", "val"] = "val",
    ):
        super().__init__(
            name=name,
            update_in=update_in,
            log_on="epoch",
        )
        self.bins = bins
        self.values_field = values_field

        self.clear()  # initialize all state values

    def _get_values(self, data) -> torch.Tensor:
        return data[self.values_field]

    def update(self, data):
        values = self._get_values(data).flatten()
        hist, _ = torch.histogram(values.float(), bins=self.bins, density=False)
        self.num += len(values)
        self.totals += hist
        self.min = min(self.min, values.min().item())
        self.max = max(self.max, values.max().item())
        self.sum += values.sum()
        self.sum_square += values.dot(values)

    def log(self, loggers: list, current_epoch: int, batch_idx: int):
        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                experiment: SummaryWriter = logger.experiment
                experiment.add_histogram_raw(
                    self.name,
                    min=self.min,
                    max=self.max,
                    num=self.num,
                    sum=self.sum,
                    sum_squares=self.sum_square,
                    bucket_limits=self.bins[1:],
                    bucket_counts=self.totals,
                    global_step=current_epoch,
                )

    def clear(self):
        self.totals = torch.zeros(len(self.bins) - 1)
        self.num = 0
        self.sum = 0
        self.sum_square = 0
        self.min = math.inf
        self.max = -math.inf


class MultiLabelSegmentation(SegmentationTaskMixin, Task):
    """Generic multi-label segmentation

    Multi-label segmentation is the process of detecting temporal intervals
    when a specific audio class is active.

    Example use cases include speaker tracking, gender (male/female)
    classification, or audio event detection.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    classes : List[str], optional
        List of classes. Defaults to the list of classes available in the training set.
    duration : float, optional
        Chunks duration. Defaults to 2s.
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
    weight: str, optional
        When provided, use this key to as frame-wise weight in loss function.
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
        Defaults to AUROC (area under the ROC curve).
    """

    def __init__(
        self,
        protocol: Protocol,
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        metric_classwise: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        loggables: List[Loggable] = None,
    ):

        if not isinstance(protocol, SegmentationProtocol):
            raise ValueError(
                f"MultiLabelSegmentation task expects a SegmentationProtocol but you gave {type(protocol)}. "
            )

        super().__init__(
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

        self.balance = balance
        self.weight = weight
        self.classes = classes
        self._metric_classwise = metric_classwise

        if loggables is None:
            loggables = []
        elif isinstance(loggables, Loggable):
            loggables = [loggables]
        self.loggables = loggables

        # task specification depends on the data: we do not know in advance which
        # classes should be detected. therefore, we postpone the definition of
        # specifications to setup()

    def setup(self, stage: Optional[str] = None):

        super().setup(stage=stage)

        self.specifications = Specifications(
            classes=self.classes,
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
        )

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk for multi-label segmentation

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target (see Notes below)
            - `meta`:
                - `database`: database index
                - `file`: file index

        Notes
        -----
        y is a trinary matrix with shape (num_frames, num_classes):
            -  0: class is inactive
            -  1: class is active
            - -1: we have no idea

        """

        file = self.get_file(file_id)

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # TODO: this should be cached
        # use model introspection to predict how many frames it will output
        num_samples = sample["X"].shape[1]
        num_frames, _ = self.model.introspection(num_samples)
        resolution = duration / num_frames
        frames = SlidingWindow(start=0.0, duration=resolution, step=resolution)

        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        start_idx = np.floor(start / resolution).astype(int)
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / resolution).astype(int)

        # frame-level targets (-1 for un-annotated classes)
        y = -np.ones((num_frames, len(self.classes)), dtype=np.int8)
        y[:, self.annotated_classes[file_id]] = 0
        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations["global_label_idx"]
        ):
            y[start:end, label] = 1

        sample["y"] = SlidingWindowFeature(y, frames, labels=self.classes)

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def training_step(self, batch, batch_idx: int):

        X = batch["X"]
        y_pred = self.model(X)
        y_true = batch["y"]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = F.binary_cross_entropy(y_pred, y_true.type(torch.float))

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            f"{self.logging_prefix}TrainLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):

        X = batch["X"]
        y_pred = self.model(X)
        y_true = batch["y"]
        assert y_pred.shape == y_true.shape

        # TODO: add support for frame weights
        # TODO: add support for class weights

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred_labelled = y_pred[mask]
        y_true_labelled = y_true[mask]
        loss = F.binary_cross_entropy(
            y_pred_labelled, y_true_labelled.type(torch.float)
        )

        # log global metric
        # TODO: allow using real multilabel metrics (when mask.all() ?)
        self.model.validation_metric(
            y_pred_labelled,
            y_true_labelled,
        )
        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        loggable_data = {
            "X": X,
            "y_pred": y_pred_labelled,
            "y_true": y_true_labelled,
        }
        for loggable in self.loggables:
            if loggable.update_in == "val":
                loggable.update(loggable_data)
            if loggable.compute_on == "step":
                loggable.log(self.model.loggers, self.model.current_epoch, batch_idx)
                loggable.clear()
        # if batch_idx == 0:
        #     for logger in self.model.loggers:
        #         if isinstance(logger, TensorBoardLogger):
        #             experiment: SummaryWriter = logger.experiment

        #             bins = torch.linspace(0, 1, 15 + 1)
        #             values = torch.rand(50000)
        #             hist, _ = torch.histogram(values, bins=bins, density=False)
        #             sum_sq = values.dot(values)

        #             experiment.add_histogram_raw(
        #                 f"{self.logging_prefix}HistogramTest",
        #                 min=values.min(),
        #                 max=values.max(),
        #                 num=len(values),
        #                 sum=values.sum(),
        #                 sum_squares=sum_sq,
        #                 bucket_limits=bins[1:],
        #                 bucket_counts=hist,
        #                 global_step=self.model.current_epoch,
        #             )
        #             print("logged histogram :D")

        # log metrics per class
        for class_id, class_name in enumerate(self.classes):
            mask: torch.Tensor = y_true[..., class_id] != -1
            if mask.sum() == 0:
                continue

            y_pred_labelled = y_pred[..., class_id][mask]
            y_true_labelled = y_true[..., class_id][mask]

            metric = self.model.validation_metric_classwise[class_name]
            metric(
                y_pred_labelled,
                y_true_labelled,
            )

            self.model.log_dict(
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.model.log(
            f"{self.logging_prefix}ValLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def on_validation_end(self):
        super().on_validation_end()
        print("on_validation_end :D")
        for loggable in self.loggables:
            if loggable.compute_on == "epoch":
                loggable.log(self.model.loggers, self.model.current_epoch, -1)
                loggable.clear()

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        class_count = len(self.classes)
        if class_count > 1:  # multilabel
            # task is binary because in case some targets are missing, we
            # can't compute multilabel metrics anymore (torchmetrics doesn't allow
            # us to ignore specific classes for specific data points)
            return [
                F1Score(task="binary"),
                Precision(task="binary"),
                Recall(task="binary"),
            ]
        else:
            # This case is handled by the per-class metric, see 'default_metric_classwise'
            return []

    def default_metric_classwise(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return [
            F1Score(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
            CalibrationError(task="binary"),
        ]

    @cached_property
    def metric_classwise(self) -> MetricCollection:
        if self._metric_classwise is None:
            self._metric_classwise = self.default_metric_classwise()

        return MetricCollection(self._metric_classwise, prefix=self.logging_prefix)

    def setup_validation_metric(self):
        # setup validation metric
        super().setup_validation_metric()

        # and then setup validation metric per class
        metric = self.metric_classwise
        if metric is None:
            return

        self.model.validation_metric_classwise = torch.nn.ModuleDict().to(
            self.model.device
        )
        for class_name in self.classes:
            self.model.validation_metric_classwise[class_name] = metric.clone(
                prefix=self.logging_prefix, postfix=f"-{class_name}"
            )

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

        return f"{self.logging_prefix}ValLoss", "min"
