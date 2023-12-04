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

import itertools
import multiprocessing
import numpy as np
from pathlib import Path
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from functools import cached_property, partial
from numbers import Number
from typing import Dict, List, Literal, Optional, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import scipy.special
import torch
from pyannote.database import Protocol
from pyannote.database.protocol import SegmentationProtocol, SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import Scope, Subset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_audiomentations import Identity
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, MetricCollection

from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.protocol import check_protocol

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)

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
    cache_path : str, optional
       File path where store task-related data, especially data from protocol

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
        cache_path: Optional[Union[str, None]] = None
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
        self.cache_path = cache_path
        self.prepared_data = {}

    def prepare_data(self):
        """Use this to download and prepare data

        This is where we might end up downloading datasets
        and transform them so that they are ready to be used
        with pyannote.database. but for now, the API assume
        that we directly provide a pyannote.database.Protocol.

        Notes
        -----
        Called only once on the main process (and only on it), for global_rank 0.
        """

        def get_smallest_type(value: int, unsigned: Optional[bool]=False) -> str:
            """Return the most suitable type for storing the
            value passed in parameter in memory.

            Parameters
            ----------
            value: int
                value whose type is best suited to storage in memory
            unsigned: bool, optional
                positive integer mode only. Default to False
            Returns
            -------
            str:
                numpy formatted type
                (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
            """
            if unsigned:
                if value < 0:
                    raise ValueError(
                        f"negative value ({value}) is incompatible with unsigned types"
                    )
                # unsigned byte (8 bits), unsigned short (16 bits), unsigned int (32 bits)
                types_list = [(255, 'B'), (65_535, 'u2'), (4_294_967_296, 'u4')]
            else:
                # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
                types_list = [(127, 'b'), (32_768, 'i2'), (2_147_483_648, 'i')]
            filtered_list = [(max_val, type) for max_val, type in types_list if max_val > abs(value)]
            if not filtered_list:
                return 'u8' if unsigned else 'i8' # unsigned or signed long (64 bits)
            return filtered_list[0][1]

        if self.cache_path is not None:
            cache_path = Path(self.cache_path)
            if cache_path.exists():
                # data was already created, nothing to do
                return
            # create a new cache directory at the path specified by the user
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        # duration of training chunks
        # TODO: handle variable duration case
        duration = getattr(self, "duration", 0.0)

        # list of possible values for each metadata key
        metadata_unique_values = defaultdict(list)

        metadata_unique_values["subset"] = Subsets

        if isinstance(self.protocol, SpeakerDiarizationProtocol):
            metadata_unique_values["scope"] = Scopes

        elif isinstance(self.protocol, SegmentationProtocol):
            classes = getattr(self, "classes", list())

        # save all protocol data in a dict
        prepared_data = {}

        prepared_data["protocol_name"] = self.protocol.name
        # make sure classes attribute exists (and set to None if it did not exist)
        prepared_data["classes"] = getattr(self, "classes", None)
        if prepared_data["classes"] is None:
            classes = list()
            # metadata_unique_values["classes"] = list(classes)
        else:
            classes = prepared_data["classes"]

        audios = list()  # list of path to audio files
        audio_infos = list()
        audio_encodings = list()
        metadata = list()  # list of metadata

        annotated_duration = list()  # total duration of annotated regions (per file)
        annotated_regions = list()  # annotated regions
        annotations = list()  # actual annotations
        annotated_classes = list()  # list of annotated classes (per file)
        unique_labels = list()

        if self.has_validation:
            files_iter = itertools.chain(
                self.protocol.train(), self.protocol.development()
            )
        else:
            files_iter = self.protocol.train()

        for file_id, file in enumerate(files_iter):
            # gather metadata and update metadata_unique_values so that each metadatum
            # (e.g. source database or label) is represented by an integer.
            metadatum = dict()

            # keep track of source database and subset (train, development, or test)
            if file["database"] not in metadata_unique_values["database"]:
                metadata_unique_values["database"].append(file["database"])
            metadatum["database"] = metadata_unique_values["database"].index(
                file["database"]
            )
            metadatum["subset"] = Subsets.index(file["subset"])

            # keep track of speaker label scope (file, database, or global) for speaker diarization protocols
            if isinstance(self.protocol, SpeakerDiarizationProtocol):
                metadatum["scope"] = Scopes.index(file["scope"])

            # keep track of list of classes for regular segmentation protocols
            # Different files may be annotated using a different set of classes
            # (e.g. one database for speech/music/noise, and another one for male/female/child)
            if isinstance(self.protocol, SegmentationProtocol):
                if "classes" in file:
                    local_classes = file["classes"]
                else:
                    local_classes = file["annotation"].labels()

                # if task was not initialized with a fixed list of classes,
                # we build it as the union of all classes found in files
                if prepared_data["classes"] is None:
                    for klass in local_classes:
                        if klass not in classes:
                            classes.append(klass)
                    annotated_classes.append(
                        [classes.index(klass) for klass in local_classes]
                    )

                # if task was initialized with a fixed list of classes,
                # we make sure that all files use a subset of these classes
                # if they don't, we issue a warning and ignore the extra classes
                else:
                    extra_classes = set(local_classes) - set(prepared_data["classes"])
                    if extra_classes:
                        warnings.warn(
                            f"Ignoring extra classes ({', '.join(extra_classes)}) found for file {file['uri']} ({file['database']}). "
                        )
                    annotated_classes.append(
                        [
                            prepared_data["classes"].index(klass)
                            for klass in set(local_classes) & set(prepared_data["classes"])
                        ]
                    )

            remaining_metadata_keys = set(file) - set(
                [
                    "uri",
                    "database",
                    "subset",
                    "audio",
                    "torchaudio.info",
                    "scope",
                    "classes",
                    "annotation",
                    "annotated",
                ]
            )

            # keep track of any other (integer or string) metadata provided by the protocol
            # (e.g. a "domain" key for domain-adversarial training)
            for key in remaining_metadata_keys:
                value = file[key]

                if isinstance(value, str):
                    if value not in metadata_unique_values[key]:
                        metadata_unique_values[key].append(value)
                    metadatum[key] = metadata_unique_values[key].index(value)

                elif isinstance(value, int):
                    metadatum[key] = value

                else:
                    warnings.warn(
                        f"Ignoring '{key}' metadata because of its type ({type(value)}). Only str and int are supported for now.",
                        category=UserWarning,
                    )

            metadata.append(metadatum)

            database_unique_labels = list()

            # reset list of file-scoped labels
            file_unique_labels = list()

            # path to audio file
            audios.append(str(file["audio"]))

            # audio info
            audio_info = file["torchaudio.info"]
            audio_infos.append(
                (
                    audio_info.sample_rate,  # sample rate
                    audio_info.num_frames,  # number of frames
                    audio_info.num_channels,  # number of channels
                    audio_info.bits_per_sample,  # bits per sample
                )
            )
            audio_encodings.append(audio_info.encoding)  # encoding

            # annotated regions and duration
            _annotated_duration = 0.0
            for segment in file["annotated"]:
                # skip annotated regions that are shorter than training chunk duration
                if segment.duration < duration:
                    continue

                # append annotated region
                annotated_region = (
                    file_id,
                    segment.duration,
                    segment.start,
                )
                annotated_regions.append(annotated_region)

                # increment annotated duration
                _annotated_duration += segment.duration

            # append annotated duration
            annotated_duration.append(_annotated_duration)

            # annotations
            for segment, _, label in file["annotation"].itertracks(yield_label=True):
                # "scope" is provided by speaker diarization protocols to indicate
                # whether speaker labels are local to the file ('file'), consistent across
                # all files in a database ('database'), or globally consistent ('global')

                if "scope" in file:
                    # 0 = 'file'
                    # 1 = 'database'
                    # 2 = 'global'
                    scope = Scopes.index(file["scope"])

                    # update list of file-scope labels
                    if label not in file_unique_labels:
                        file_unique_labels.append(label)
                    # and convert label to its (file-scope) index
                    file_label_idx = file_unique_labels.index(label)

                    database_label_idx = global_label_idx = -1

                    if scope > 0:  # 'database' or 'global'
                        # update list of database-scope labels
                        if label not in database_unique_labels:
                            database_unique_labels.append(label)

                        # and convert label to its (database-scope) index
                        database_label_idx = database_unique_labels.index(label)

                    if scope > 1:  # 'global'
                        # update list of global-scope labels
                        if label not in unique_labels:
                            unique_labels.append(label)
                        # and convert label to its (global-scope) index
                        global_label_idx = unique_labels.index(label)

                # basic segmentation protocols do not provide "scope" information
                # as classes are global by definition

                else:
                    try:
                        file_label_idx = (
                            database_label_idx
                        ) = global_label_idx = classes.index(label)
                    except ValueError:
                        # skip labels that are not in the list of classes
                        continue

                annotations.append(
                    (
                        file_id,  # index of file
                        segment.start,  # start time
                        segment.end,  # end time
                        file_label_idx,  # file-scope label index
                        database_label_idx,  # database-scope label index
                        global_label_idx,  # global-scope index
                    )
                )

        # since not all metadata keys are present in all files, fallback to -1 when a key is missing
        metadata = [
            tuple(metadatum.get(key, -1) for key in metadata_unique_values)
            for metadatum in metadata
        ]
        dtype = [
            (key, get_smallest_type(max(m[i] for m in metadata))) for i, key in enumerate(metadata_unique_values)
        ]

        prepared_data["metadata"] = np.array(metadata, dtype=dtype)
        metadata.clear()
        prepared_data["audios"] = np.array(audios, dtype=np.string_)
        audios.clear()

        # turn list of files metadata into a single numpy array
        # TODO: improve using https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
        dtype = [
            ("sample_rate", get_smallest_type(max(ai[0] for ai in audio_infos), unsigned=True)),
            ("num_frames",  get_smallest_type(max(ai[1] for ai in audio_infos), unsigned=True)),
            ("num_channels", "B"),
            ("bits_per_sample", "B"),
        ]
        prepared_data["audio_infos"] = np.array(audio_infos, dtype=dtype)
        audio_infos.clear()
        prepared_data["audio_encodings"] = np.array(audio_encodings, dtype=np.string_)
        audio_encodings.clear()
        prepared_data["annotated_duration"] = np.array(annotated_duration)
        annotated_duration.clear()

        # turn list of annotated regions into a single numpy array
        dtype = [
            ("file_id", get_smallest_type(max(ar[0] for ar in annotated_regions), unsigned=True)),
            ("duration", "f"),
            ("start", "f")
        ]
        prepared_data["annotated_regions"] = np.array(annotated_regions, dtype=dtype)
        annotated_regions.clear()

        # convert annotated_classes (which is a list of list of classes, one list of classes per file)
        # into a single (num_files x num_classes) numpy array:
        #    * True indicates that this particular class was annotated for this particular file (though it may not be active in this file)
        #    * False indicates that this particular class was not even annotated (i.e. its absence does not imply that it is not active in this file)
        if isinstance(self.protocol, SegmentationProtocol) and prepared_data["classes"] is None:
            prepared_data["classes"] = classes
        annotated_classes_array = np.zeros(
            (len(annotated_classes), len(classes)), dtype=np.bool_
        )
        for file_id, classes in enumerate(annotated_classes):
            annotated_classes_array[file_id, classes] = True
        prepared_data["annotated_classes"] = annotated_classes_array
        annotated_classes.clear()

        # turn list of annotations into a single numpy array
        dtype = [
            ("file_id", get_smallest_type(max(a[0] for a in annotations), unsigned=True)),
            ("start", "f"),
            ("end", "f"),
            ("file_label_idx", get_smallest_type(max(a[3] for a in annotations))),
            ("database_label_idx", get_smallest_type(max(a[4] for a in annotations))),
            ("global_label_idx", get_smallest_type(max(a[5] for a in annotations))),
        ]

        prepared_data["annotations"] = np.array(annotations, dtype=dtype)
        annotations.clear()
        prepared_data["metadata_unique_values"] = metadata_unique_values

        if self.has_validation:
            validation_chunks = list()

            # obtain indexes of files in the validation subset
            validation_file_ids = np.where(
                prepared_data["metadata"]["subset"] == Subsets.index("development")
            )[0]

            # iterate over files in the validation subset
            for file_id in validation_file_ids:
                # get annotated regions in file
                annotated_regions = prepared_data["annotated_regions"][
                    prepared_data["annotated_regions"]["file_id"] == file_id
                ]

                # iterate over annotated regions
                for annotated_region in annotated_regions:
                    # number of chunks in annotated region
                    num_chunks = round(annotated_region["duration"] // duration)

                    # iterate over chunks
                    for c in range(num_chunks):
                        start_time = annotated_region["start"] + c * duration
                        validation_chunks.append((file_id, start_time, duration))

            dtype = [
                ("file_id", get_smallest_type(max(v[0] for v in validation_chunks), unsigned=True)),
                ("start", "f"),
                ("duration", "f")
            ]
            prepared_data["validation_chunks"] = np.array(validation_chunks, dtype=dtype)
            validation_chunks.clear()

        self.prepared_data = prepared_data
        self.has_setup_metadata = True

        # save preparated data on the disk
        if self.cache_path is not None:
            with open(self.cache_path, 'wb') as cache_file:
                np.savez_compressed(cache_file, **prepared_data)

        self.has_prepared_data = True

    def setup(self, stage=None):
        """Setup data on each device"""
        # if all data was assigned to the task, nothing to do
        if self.has_setup_metadata:
            return
        # if no cache directory was provided by the user and task data was not already prepared
        if self.cache_path is None and not self.has_prepared_data:
            warnings.warn("No path to the directory containing the cache of prepared data"
                            " has been specified. Data preparation will therefore be carried out"
                            " on each process used for training. To speed up data preparation, you"
                            " can specify a cache directory when instantiating the task.", stacklevel=1)
            self.prepare_data()
            return
        # load data cached by prepare_data method into the task:
        try:
            with open(self.cache_path, 'rb') as cache_file:
                self.prepared_data = dict(np.load(cache_file, allow_pickle=True))
        except FileNotFoundError:
            print("""Cached data for protocol not found. Ensure that prepare_data was
                    executed correctly and that the path to the task cache is correct""")
            raise
        # checks that the task current protocol matches the cached protocol
        if self.protocol.name != self.prepared_data["protocol_name"]:
            raise ValueError(
                f"Protocol specified for the task ({self.protocol.name}) "
                f"does not correspond to the cached one ({self.prepared_data['protocol_name']})"
            )
        self.has_setup_metadata = True


    @property
    def specifications(self) -> Union[Specifications, Tuple[Specifications]]:
        # setup metadata on-demand the first time specifications are requested and missing
        if not hasattr(self, "_specifications"):
            self.setup()
        return self._specifications

    @specifications.setter
    def specifications(
        self, specifications: Union[Specifications, Tuple[Specifications]]
    ):
        self._specifications = specifications

    @property
    def has_prepared_data(self):
        # This flag indicates if data for this task was generated, and
        # optionally saved on the disk
        return getattr(self, "_has_prepared_data", False)

    @has_prepared_data.setter
    def has_prepared_data(self, value: bool):
        self._has_setup_metadata = value

    @property
    def has_setup_metadata(self):
        # This flag indicates if data was assigned to this task, directly from prepared
        # data or by reading in a cached file on the disk
        return getattr(self, "_has_setup_metadata", False)

    @has_setup_metadata.setter
    def has_setup_metadata(self, value: bool):
        self._has_setup_metadata = value

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
