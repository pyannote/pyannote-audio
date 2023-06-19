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

from collections import defaultdict
import itertools
import random
import numpy as np
import torch
from typing import Literal, List, Text, Union, Sequence, Dict
import warnings

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchaudio.backend.common import AudioMetaData
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC
from torch.utils.data._utils.collate import default_collate

from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.database.protocol import SegmentationProtocol, SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import Scope, Subset
from pyannote.audio.torchmetrics.classification import EqualErrorRate
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)

Subtask = Literal["diarization", "embedding"]

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)
Subtasks = list(Subtask.__args__)


class JointSpeakerDiarizationAndEmbedding(Task):
    """Joint speaker diarization and embedding task

    Usage
    -----
    load a meta protocol containing both diarization (e.g. X.SpeakerDiarization.Pretraining)
    and verification (e.g. VoxCeleb.SpeakerVerification.VoxCeleb) datasets
    >>> from pyannote.database import registry
    >>> protocol = registry.get_protocol(...)

    instantiate task
    >>> task = JointSpeakerDiarizationAndEmbedding(protocol)

    instantiate multi-task model
    >>> model = JointSpeakerDiarizationAndEmbeddingModel()
    >>> model.task = task

    train as usual...

    """

    def __init__(
            self,
            protocol,
            duration: float = 5.0,
            max_speakers_per_chunk: int = 3,
            max_speakers_per_frame: int = 2,
            batch_size: int = 32,
            database_ratio : float = 0.5,
            num_workers: int = None,
            pin_memory: bool = False,
            augmentation: BaseWaveformTransform = None
    ) -> None:
        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.database_ratio = database_ratio


        # keep track of the use of database available in the meta protocol
        # * embedding databases are those with global speaker label scope
        # * diarization databases are those with file or database speaker label scope
        self.embedding_database_files = []
        self.diarization_database_files = []

    def get_file(self, file_id):

        file = dict()

        file["audio"] = str(self.audios[file_id], encoding="utf-8")

        _audio_info = self.audio_infos[file_id]
        _encoding = self.audio_encodings[file_id]

        sample_rate = _audio_info["sample_rate"]
        num_frames = _audio_info["num_frames"]
        num_channels = _audio_info["num_channels"]
        bits_per_sample = _audio_info["bits_per_sample"]
        encoding = str(_encoding, encoding="utf-8")
        file["torchaudio.info"] = AudioMetaData(
            sample_rate=sample_rate,
            num_frames=num_frames,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            encoding=encoding,
        )

        return file

    def setup(self, stage="fit"):
        """Setup method

        Parameters
        ----------
        stage : {'fit', 'validate', 'test'}, optional
            Setup stage. Defaults to 'fit'.
        """

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

        # make sure classes attribute exists (and set to None if it did not exist)
        self.classes = getattr(self, "classes", None)
        if self.classes is None:
            classes = list()
            # metadata_unique_values["classes"] = list(classes)

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
                # add the file to the embedding or diarization list according to the file database speaker
                # labels scope
                if file["scope"] == 'global':
                    self.embedding_database_files.append(file_id)
                elif file["scope"] in ["database", "file"]:
                    self.diarization_database_files.append(file_id)

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
                if self.classes is None:
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
                    extra_classes = set(local_classes) - set(self.classes)
                    if extra_classes:
                        warnings.warn(
                            f"Ignoring extra classes ({', '.join(extra_classes)}) found for file {file['uri']} ({file['database']}). "
                        )
                    annotated_classes.append(
                        [
                            self.classes.index(klass)
                            for klass in set(local_classes) & set(self.classes)
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
                    segment.end,
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
        dtype = [(key, "i") for key in metadata_unique_values]
        self.metadata = np.array(metadata, dtype=dtype)

        # NOTE: read with str(self.audios[file_id], encoding='utf-8')
        self.audios = np.array(audios, dtype=np.string_)

        # turn list of files metadata into a single numpy array
        # TODO: improve using https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519

        dtype = [
            ("sample_rate", "i"),
            ("num_frames", "i"),
            ("num_channels", "i"),
            ("bits_per_sample", "i"),
        ]
        self.audio_infos = np.array(audio_infos, dtype=dtype)
        self.audio_encodings = np.array(audio_encodings, dtype=np.string_)

        self.annotated_duration = np.array(annotated_duration)

        # turn list of annotated regions into a single numpy array
        dtype = [("file_id", "i"), ("duration", "f"), ("start", "f"), ("end", "f")]
        self.annotated_regions = np.array(annotated_regions, dtype=dtype)

        # convert annotated_classes (which is a list of list of classes, one list of classes per file)
        # into a single (num_files x num_classes) numpy array:
        #    * True indicates that this particular class was annotated for this particular file (though it may not be active in this file)
        #    * False indicates that this particular class was not even annotated (i.e. its absence does not imply that it is not active in this file)
        if isinstance(self.protocol, SegmentationProtocol) and self.classes is None:
            self.classes = classes
        self.annotated_classes = np.zeros(
            (len(annotated_classes), len(self.classes)), dtype=np.bool_
        )
        for file_id, classes in enumerate(annotated_classes):
            self.annotated_classes[file_id, classes] = True

        # turn list of annotations into a single numpy array
        dtype = [
            ("file_id", "i"),
            ("start", "f"),
            ("end", "f"),
            ("file_label_idx", "i"),
            ("database_label_idx", "i"),
            ("global_label_idx", "i"),
        ]
        self.annotations = np.array(annotations, dtype=dtype)

        self.metadata_unique_values = metadata_unique_values

        if not self.has_validation:
            return

        validation_chunks = list()

        # obtain indexes of files in the validation subset
        validation_file_ids = np.where(
            self.metadata["subset"] == Subsets.index("development")
        )[0]

        # iterate over files in the validation subset
        for file_id in validation_file_ids:

            # get annotated regions in file
            annotated_regions = self.annotated_regions[
                self.annotated_regions["file_id"] == file_id
            ]

            # iterate over annotated regions
            for annotated_region in annotated_regions:

                # number of chunks in annotated region
                num_chunks = round(annotated_region["duration"] // duration)

                # iterate over chunks
                for c in range(num_chunks):
                    start_time = annotated_region["start"] + c * duration
                    validation_chunks.append((file_id, start_time, duration))

        dtype = [("file_id", "i"), ("start", "f"), ("duration", "f")]
        self.validation_chunks = np.array(validation_chunks, dtype=dtype)

        speaker_diarization = Specifications(
                duration=self.duration,
                resolution=Resolution.FRAME,
                problem=Problem.MONO_LABEL_CLASSIFICATION,
                permutation_invariant=True,
                classes=[f"speaker{i+1}" for i in range(self.max_speakers_per_chunk)],
                powerset_max_classes=self.max_speakers_per_frame,
            )
        speaker_embedding = Specifications(
            duration=self.duration,
            resolution=Resolution.CHUNK,
            problem=Problem.REPRESENTATION,
            classes=unique_labels,
        )

        self.specifications = (speaker_diarization, speaker_embedding)

    def prepare_chunk(self, file_id: int, start_time: float, duration: float, subtask: int):
        """Prepare chunk

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.
        subtask: int
            - 0 : diarization task
            - 1 : embedding task

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target as a SlidingWindowFeature instance where y.labels is
                   in meta.scope space.
            - `meta`:
                - `scope`: target scope (0: file, 1: database, 2: global)
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        # get label scope
        label_scope = Scopes[self.metadata[file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        #
        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        # TODO handle tuple outputs from the model
        start_idx = np.floor(start / self.model.example_output[0].frames.step).astype(int)
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / self.model.example_output[0].frames.step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        y = np.zeros((self.model.example_output[0].num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start:end, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.example_output[subtask].frames, labels=labels
        )

        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id
        sample["meta"]["subtask"] = subtask

        return sample

    def draw_diarization_chunk(self, file_ids : np.ndarray,
                                 prob_annotated_duration : np.ndarray,
                                 rng : random.Random,
                                 duration : float,
                                 ) -> tuple:
        """Sample one chunk for the diarization task

        Parameters
        ----------
        file_ids: np.ndarray
            array containing files id
        prob_annotated_duration: np.ndarray
            array of the same size than file_ids array, containing probability
            to corresponding file to be drawn
        rng : random.Random
            Random number generator
        duration: float
            duration of the chunk to draw
        """
        # select one file at random (wiht probability proportional to its annotated duration)
        file_id = np.random.choice(file_ids, p=prob_annotated_duration)
        # find indices of annotated regions in this file
        annotated_region_indices = np.where(
            self.annotated_regions["file_id"] == file_id
        )[0]

        # turn annotated regions duration into a probability distribution
        prob_annotaded_regions_duration = self.annotated_regions["duration"][
            annotated_region_indices
        ] / np.sum(self.annotated_regions["duration"][annotated_region_indices])

        # seletect one annotated region at random (with probability proportional to its duration)
        annotated_region_index = np.random.choice(annotated_region_indices,
                                                  p=prob_annotaded_regions_duration
                                                  )

        # select one chunk at random in this annotated region
        _, _, start, end = self.annotated_regions[annotated_region_index]
        start_time = rng.uniform(start, end - duration)

        return (file_id, start_time)

    def draw_embedding_chunk(self, klass : Text,
                               classes : List[Text],
                               duration : float) -> tuple:
        """Sample one chunk for the embedding task

        Parameters
        ----------
        klass: Text
            current class of speakers from which to draw a sample
        classes: List[Text]
            list of all the global speaker labels, in the same order than the list
            defined in the task specification
        duration: float
            duration of the chunk to draw

        Return
        ------
        tuple:
            file_id:
                the file id to which the sampled chunk belongs
            start_time:
                start time of the sampled chunk
        """
        # get index of the current class in the order of original class list
        class_id = classes.index(klass)
        # get segments for current class
        class_segments_idx = self.annotations["global_label_idx"] == class_id
        class_segments = self.annotations[class_segments_idx]

        # sample one segment from all the class segments:
        segments_duration = class_segments["end"] - class_segments["start"]
        segments_total_duration = np.sum(segments_duration)
        prob_segments = segments_duration / segments_total_duration
        segment = np.random.choice(class_segments, p=prob_segments)

        # sample chunk start time in order to intersect it with the sampled segment
        start_time = np.random.uniform(segment["start"] - duration / 2, segment["start"])

        return (segment["file_id"], start_time)

    def train__iter__helper(self, rng : random.Random, **filters):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key : value} dict), filter training files so that
            only file such as file [key] == value are used for generating chunks

        Yields
        ------
        chunk : dict
        Training chunks
        """

        # indices of trainijng files that matches domain filters
        training = self.metadata["subset"] == Subsets.index("train")
        for key, value in filters.items():
            training &= self.metadata[key] == value
        file_ids = np.where(training)[0]
        # get the subset of embedding database files from training files
        embedding_files_ids = file_ids[np.in1d(file_ids, self.embedding_database_files)]

        annotated_duration = self.annotated_duration[file_ids]
        prob_annotated_duration = annotated_duration / np.sum(annotated_duration)
        # set probability to sample a file from embedding database to 0
        prob_annotated_duration[embedding_files_ids] = 0

        duration = self.duration

        # make a copy of the original classes list, in order to not modify it when shuffling
        embedding_classes = self.specifications[Subtasks.index("embedding")].classes
        shuffled_embedding_classes = list(embedding_classes)
        embedding_class_idx = 0

        while True:
            # choose between diarization or embedding subtask according to a ratio
            # between these two tasks
            if np.random.uniform() < self.database_ratio:
                subtask = Subtasks.index("diarization")
                file_id, start_time = self.draw_diarization_chunk(file_ids, prob_annotated_duration, rng, duration)
            else:
                subtask = Subtasks.index("embedding")
                # shuffle embedding classes list and go through this shuffled list
                # to make sure to see all the speakers during training
                if embedding_class_idx == len(shuffled_embedding_classes):
                    rng.shuffle(shuffled_embedding_classes)
                    embedding_class_idx = 0
                klass = shuffled_embedding_classes[embedding_class_idx]
                embedding_class_idx += 1
                file_id, start_time = self.draw_embedding_chunk(klass,
                                                                classes=embedding_classes,
                                                                duration=duration)

            sample = self.prepare_chunk(file_id, start_time, duration, subtask)
            yield sample

    def train__iter__(self):
        """Iterate over trainig samples

        Yields
        ------
        dict:
            x: (time, channel)
                Audio chunks.
            task: "diarization" or "embedding"
            y: target speaker label for speaker embedding task,
                (frame, ) frame-level targets for speaker diarization task.
                Note that frame < time.
                `frame is infered automagically from the exemple model output`
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.model.current_epoch)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)
        else:
            # create
            subchunks = dict()
            for product in itertools.product([self.metadata_unique_values[key] for key in balance]):
                filters = {key : value for key, value in zip(balance, product)}
                subchunks[product] = self.train__iter__helper(rng, **filters)

        while True:
            # select one subchunck generator at random (with uniform probability)
            # so thath it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(subchunks)]

            # generate random chunk
            yield next(chunks)

    def collate_X(self, batch) -> torch.Tensor:
        """Collate for data"""
        return default_collate([b["X"] for b in batch])

    def collate_y(self, batch) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : list
            List of samples to collate.
            "y" field is expected to be a SlidingWindowFeature.

        Returns
        -------
        y : torch.Tensor
            Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
            If one chunk has more than `self.max_speakers_per_chunk` speakers, we keep
            the max_speakers_per_chunk most talkative ones. If it has less, we pad with
            zeros (artificial inactive speakers).
        """

        collated_y = []
        for b in batch:
            y = b["y"].data
            num_speakers = len(b["y"].labels)
            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y, axis=0), axis=0)
                # keep only the most talkative speakers
                y = y[:, indices[: self.max_speakers_per_chunk]]

                # TODO: we should also sort the speaker labels in the same way

            elif num_speakers < self.max_speakers_per_chunk:
                # create inactive speakers by zero padding
                y = np.pad(
                    y,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            else:
                # we have exactly the right number of speakers
                pass

            collated_y.append(y)

        return torch.from_numpy(np.stack(collated_y))

    def collate_meta(self, batch) -> torch.Tensor:
        """Collate for metadata"""
        return default_collate([b["meta"] for b in batch])

    def collate_fn(self, batch, stage="train"):
        """Collate function used for most segmentation tasks

        This function does the following:
        * stack waveforms into a (batch_size, num_channels, num_samples) tensor batch["X"])
        * apply augmentation when in "train" stage
        * convert targets into a (batch_size, num_frames, num_classes) tensor batch["y"]
        * collate any other keys that might be present in the batch using pytorch default_collate function

        Parameters
        ----------
        batch : list of dict
            List of training samples.

        Returns
        -------
        batch : dict
            Collated batch as {"X": torch.Tensor, "y": torch.Tensor} dict.
        """

        # collate X
        collated_X = self.collate_X(batch)

        # collate y
        try:
            collated_y = self.collate_y(batch)
        except RuntimeError as e:
            print(e)
            print([b["y"].data for b in batch])

        # collate metadata
        collated_meta = self.collate_meta(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y.unsqueeze(1),
        )

        return {
            "X": augmented.samples,
            "y": augmented.targets.squeeze(1),
            "meta": collated_meta,
        }


    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        # `clamp_min` is needed to set non-speech weight to 1.
        class_weight = (
            torch.clamp_min(self.model.powerset.cardinality, 1.0)
            if self.weigh_by_cardinality
            else None
        )
        seg_loss = nll_loss(
            permutated_prediction,
            torch.argmax(target, dim=-1),
            class_weight=class_weight,
            weight=weight,
        )

        return seg_loss

    def setup_loss_func(self):
        diarization_spec = self.specifications[Subtasks.index("diarization")]
        self.model.powerset = Powerset(
            len(diarization_spec.classes),
            diarization_spec.powerset_max_classes,
        )

    def compute_diarization_loss(self, batch : torch.Tensor):
        """"""
        X, y = batch["X"], batch["y"]
        # drop samples that contain too many speakers
        num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
        keep : torch.Tensor = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        # TODO using variable `waveform` before assignment
        waveform = waveform[keep]

        # log effective batch size
        self.model.log(
            f"{self.logging_prefix}BatchSize",
            keep.sum(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            reduce_fx="mean",
        )
        # corner case
        if not keep.any():
            return None

        # forward pass
        prediction = self.model(waveform)
        batch_size, num_frames, _ = prediction.shape
        # (batch_size, num_frames, num_classes)
        # frames weight
        weight_key = getattr(self, "weight", None)
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

        powerset = torch.nn.functional.one_hot(
            torch.argmax(prediction, dim=-1),
            self.model.powerset.num_powerset_classes,
        ).float()
        multilabel = self.model.powerset.to_multilabel(powerset)
        permutated_target, _ = permutate(multilabel, target)
        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target.float()
        )
        seg_loss = self.segmentation_loss(
            prediction, permutated_target_powerset, weight=weight
        )

        self.model.log(
            f"{self.logging_prefix}TrainSegLoss",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        loss = seg_loss
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

    def compute_embedding_loss(self, batch : torch.Tensor):
        X, y = batch["X", batch["y"]]
        loss = self.model.loss_func(self.model(X), y)

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

    def training_step(self, batch, batch_idx: int):
        """Compute loss for the joint task

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """

        alpha = 0.5
        if batch["task"] == "diarization":
            # compute diarization loss
            diarization_loss = self.compute_diarization_loss(batch=batch)
        if batch["task"] == "embedding":
            # compute embedding loss
            embedding_loss = self.compute_embedding_loss(batch=batch)
        loss = alpha * diarization_loss + (1 - alpha) * embedding_loss

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components for diarization subtask,
        and equal error rate for the embedding part
        """
        return {
            "DiarizationErrorRate": DiarizationErrorRate(0.5),
            "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
            "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
            "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
            "EqualErrorRate": EqualErrorRate(compute_on_cpu=True, distances=False),
            "BinaryAUROC": BinaryAUROC(compute_on_cpu=True),
        }
