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

import itertools
import textwrap
from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
from pyannote.database.protocol import SegmentationProtocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks.segmentation.mixins import SegmentationTask


class MultiLabelSegmentation(SegmentationTask):
    """Generic multi-label segmentation

    Multi-label segmentation is the process of detecting temporal intervals
    when a specific audio class is active.

    Example use cases include speaker tracking, gender (male/female)
    classification, or audio event detection.

    Parameters
    ----------
    protocol : Protocol
    cache : str, optional
        As (meta-)data preparation might take a very long time for large datasets,
        it can be cached to disk for later (and faster!) re-use.
        When `cache` does not exist, `Task.prepare_data()` generates training
        and validation metadata from `protocol` and save them to disk.
        When `cache` exists, `Task.prepare_data()` is skipped and (meta)-data
        are loaded from disk. Defaults to a temporary path.
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
    balance: Sequence[Text], optional
        When provided, training samples are sampled uniformly with respect to these keys.
        For instance, setting `balance` to ["database","subset"] will make sure that each
        database & subset combination will be equally represented in the training samples.
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
        cache: Optional[Union[str, None]] = None,
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Optional[Sequence[Text]] = None,
        weight: Optional[Text] = None,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
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
            cache=cache,
        )

        self.balance = balance
        self.weight = weight
        self.classes = classes

        # task specification depends on the data: we do not know in advance which
        # classes should be detected. therefore, we postpone the definition of
        # specifications to setup()

    def post_prepare_data(self, prepared_data: Dict):
        # as different files may be annotated using a different set of classes
        # (e.g. one database for speech/music/noise, and another one for
        # male/female/child), we keep track of this information. this is used
        # to know whether a missing class is considered a negative example (0) or
        # simple an unknown example (-1)

        if self.classes is None and not self.has_classes:
            msg = textwrap.dedent(
                """
                Could not infer list of classes. Either provide a list of classes when
                instantiating the task, or make sure that the training protocol provides
                a 'classes' entry. See https://github.com/pyannote/pyannote-database#segmentation
                for more details.
                """
            )

        if self.has_validation:
            files_iter = itertools.chain(
                self.protocol.train(), self.protocol.development()
            )
        else:
            files_iter = self.protocol.train()

        if self.classes is None:
            classes = list()  # overall list of classes
            annotated_classes = list()  # list of annotated classes (per file)

            for file in files_iter:
                file_classes = file.get("classes", None)

                if not file_classes:
                    msg = textwrap.dedent(
                        f"""
                        File "{file['uri']}" (from {file['database']} database) does not
                        provide a 'classes' entry. Please make sure the corresponding
                        training protocol provides a 'classes' entry for all files. See
                        https://github.com/pyannote/pyannote-database#segmentation for more
                        details.
                        """
                    )
                    raise ValueError(msg)

                for klass in file_classes:
                    if klass not in classes:
                        classes.append(klass)
                annotated_classes.append(
                    [classes.index(klass) for klass in file_classes]
                )

            prepared_data["classes-list"] = np.array(classes, dtype=np.str_)
            self.classes = classes

        else:
            annotated_classes = list()  # list of annotated classes (per file)
            for file in files_iter:
                file_classes = file.get("classes", None)

                if not file_classes:
                    msg = textwrap.dedent(
                        f"""
                        File "{file['uri']}" (from {file['database']} database) does not
                        provide a 'classes' entry. Please make sure the corresponding
                        training protocol provides a 'classes' entry for all files. See
                        https://github.com/pyannote/pyannote-database#segmentation for more
                        details.
                        """
                    )
                    raise ValueError(msg)

                extra_classes = set(file_classes) - set(self.classes)
                if extra_classes:
                    msg = textwrap.dedent(
                        f"""
                        File "{file['uri']}" (from {file['database']} database) provides
                        extra classes ({', '.join(extra_classes)}) that are ignored.
                        """
                    )
                    print(msg)

                annotated_classes.append(
                    [
                        self.classes.index(klass)
                        for klass in set(file_classes) & set(self.classes)
                    ]
                )

            prepared_data["classes-list"] = np.array(self.classes, dtype=np.str_)

        # convert annotated_classes (which is a list of list of classes, one list of classes per file)
        # into a single (num_files x num_classes) numpy array:
        #    * True indicates that this particular class was annotated for this particular file
        #    (though it may not be active in this file)
        #    * False indicates that this particular class was not even annotated (i.e. its absence
        #    does not imply that it is not active in this file)
        annotated_classes_array = np.zeros(
            (len(annotated_classes), len(self.classes)), dtype=np.bool_
        )
        for file_id, classes in enumerate(annotated_classes):
            annotated_classes_array[file_id, classes] = True
        prepared_data["classes-annotated"] = annotated_classes_array
        annotated_classes.clear()

    def setup(self, stage=None):
        super().setup(stage)

        self.specifications = Specifications(
            classes=self.prepared_data["classes-list"],
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
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
        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start - half
        end_idx = np.round(end / step).astype(int)

        # frame-level targets (-1 for un-annotated classes)
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = -np.ones(
            (
                num_frames,
                len(self.prepared_data["classes-list"]),
            ),
            dtype=np.int8,
        )
        y[:, self.prepared_data["classes-annotated"][file_id]] = 0
        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations["global_label_idx"]
        ):
            y[start : end + 1, label] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.receptive_field, labels=self.classes
        )

        metadata = self.prepared_data["audio-metadata"][file_id]
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
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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

        # TODO: compute metrics for each class separately

        # mask (frame, class) index for which label is missing
        mask: torch.Tensor = y_true != -1
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        loss = F.binary_cross_entropy(y_pred, y_true.type(torch.float))

        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

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

        return "loss/val", "min"
