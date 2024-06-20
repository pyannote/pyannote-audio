# MIT License
#
# Copyright (c) 2024- CNRS
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

# AUTHOR: Joonas Kalda (github.com/joonaskalda)

import itertools
import math
import random
import warnings
from collections import Counter
from functools import partial
from typing import Dict, Literal, Optional, Sequence, Text, Union

import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from rich.progress import track
from torch.utils.data import DataLoader, IterableDataset
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks.segmentation.mixins import SegmentationTask, Task
from pyannote.audio.torchmetrics import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
)
from pyannote.audio.utils.loss import binary_cross_entropy
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker

try:
    from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr

    ASTEROID_IS_AVAILABLE = True
except ImportError:
    ASTEROID_IS_AVAILABLE = False


Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)


class ValDataset(IterableDataset):
    """Validation dataset class

    Val dataset needs to be iterable so that mixture of mixture generation
    can be performed in the same way for both training and development.

    Parameters
    ----------
    task : PixIT
        Task instance.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.val__iter__()

    def __len__(self):
        return self.task.val__len__()


class PixIT(SegmentationTask):
    """Joint speaker diarization and speaker separation task based on PixIT

    Parameters
    ----------
    protocol : SpeakerDiarizationProtocol
        pyannote.database protocol
    cache : str, optional
        As (meta-)data preparation might take a very long time for large datasets,
        it can be cached to disk for later (and faster!) re-use.
        When `cache` does not exist, `Task.prepare_data()` generates training
        and validation metadata from `protocol` and save them to disk.
        When `cache` exists, `Task.prepare_data()` is skipped and (meta)-data
        are loaded from disk. Defaults to a temporary path.
    duration : float, optional
        Chunks duration. Defaults to 5s.
    max_speakers_per_chunk : int, optional
        Maximum number of speakers per chunk (must be at least 2).
        Defaults to estimating it from the training set.
    max_speakers_per_frame : int, optional
        Maximum number of (overlapping) speakers per frame.
        Setting this value to 1 or more enables `powerset multi-class` training.
        Default behavior is to use `multi-label` training.
    weigh_by_cardinality: bool, optional
        Weigh each powerset classes by the size of the corresponding speaker set.
        In other words, {0, 1} powerset class weight is 2x bigger than that of {0}
        or {1} powerset classes. Note that empty (non-speech) powerset class is
        assigned the same weight as mono-speaker classes. Defaults to False (i.e. use
        same weight for every class). Has no effect with `multi-label` training.
    balance: Sequence[Text], optional
        When provided, training samples are sampled uniformly with respect to these keys.
        For instance, setting `balance` to ["database","subset"] will make sure that each
        database & subset combination will be equally represented in the training samples.
    weight: str, optional
        When provided, use this key as frame-wise weight in loss function.
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
    separation_loss_weight : float, optional
        Scaling factor between diarization and separation losses. Defaults to 0.5.
    finetune_wavlm : bool, optional
        If True, the WavLM feature extractor will be fine-tuned during training.
        Defaults to True.

    References
    ----------
    Joonas Kalda, Clément Pagés, Ricard Marxer, Tanel Alumäe, and Hervé Bredin.
    "PixIT: Joint Training of Speaker Diarization and Speech Separation
    from Real-world Multi-speaker Recordings"
    Odyssey 2024. https://arxiv.org/abs/2403.02288
    """

    def __init__(
        self,
        protocol: SpeakerDiarizationProtocol,
        cache: Optional[Union[str, None]] = None,
        duration: float = 5.0,
        max_speakers_per_chunk: Optional[int] = None,
        max_speakers_per_frame: Optional[int] = None,
        weigh_by_cardinality: bool = False,
        balance: Optional[Sequence[Text]] = None,
        weight: Optional[Text] = None,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        max_num_speakers: Optional[
            int
        ] = None,  # deprecated in favor of `max_speakers_per_chunk``
        loss: Literal["bce", "mse"] = None,  # deprecated
        separation_loss_weight: float = 0.5,
        finetune_wavlm: bool = True,
    ):
        if not ASTEROID_IS_AVAILABLE:
            raise ImportError(
                "'asteroid' must be installed to train separation models with PixIT . "
                "`pip install pyannote-audio[separation]` should do the trick."
            )

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
            cache=cache,
        )

        if not isinstance(protocol, SpeakerDiarizationProtocol):
            raise ValueError(
                "SpeakerDiarization task requires a SpeakerDiarizationProtocol."
            )

        # deprecation warnings
        if max_speakers_per_chunk is None and max_num_speakers is not None:
            max_speakers_per_chunk = max_num_speakers
            warnings.warn(
                "`max_num_speakers` has been deprecated in favor of `max_speakers_per_chunk`."
            )
        if loss is not None:
            warnings.warn("`loss` has been deprecated and has no effect.")

        # parameter validation
        if max_speakers_per_frame is not None:
            raise NotImplementedError(
                "Diarization is done on masks separately which is incompatible powerset training"
            )

        if batch_size % 2 != 0:
            raise ValueError("`batch_size` must be divisible by 2 for PixIT")

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.weigh_by_cardinality = weigh_by_cardinality
        self.balance = balance
        self.weight = weight
        self.separation_loss_weight = separation_loss_weight
        self.mixit_loss = MixITLossWrapper(multisrc_neg_sisdr, generalized=True)
        self.finetune_wavlm = finetune_wavlm

    def setup(self, stage=None):
        super().setup(stage)

        # estimate maximum number of speakers per chunk when not provided
        if self.max_speakers_per_chunk is None:
            training = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
                "train"
            )

            num_unique_speakers = []
            progress_description = f"Estimating maximum number of speakers per {self.duration:g}s chunk in the training set"
            for file_id in track(
                np.where(training)[0], description=progress_description
            ):
                annotations = self.prepared_data["annotations-segments"][
                    np.where(
                        self.prepared_data["annotations-segments"]["file_id"] == file_id
                    )[0]
                ]
                annotated_regions = self.prepared_data["annotations-regions"][
                    np.where(
                        self.prepared_data["annotations-regions"]["file_id"] == file_id
                    )[0]
                ]
                for region in annotated_regions:
                    # find annotations within current region
                    region_start = region["start"]
                    region_end = region["start"] + region["duration"]
                    region_annotations = annotations[
                        np.where(
                            (annotations["start"] >= region_start)
                            * (annotations["end"] <= region_end)
                        )[0]
                    ]

                    for window_start in np.arange(
                        region_start, region_end - self.duration, 0.25 * self.duration
                    ):
                        window_end = window_start + self.duration
                        window_annotations = region_annotations[
                            np.where(
                                (region_annotations["start"] <= window_end)
                                * (region_annotations["end"] >= window_start)
                            )[0]
                        ]
                        num_unique_speakers.append(
                            len(np.unique(window_annotations["file_label_idx"]))
                        )

            # because there might a few outliers, estimate the upper bound for the
            # number of speakers as the 97th percentile

            num_speakers, counts = zip(*list(Counter(num_unique_speakers).items()))
            num_speakers, counts = np.array(num_speakers), np.array(counts)

            sorting_indices = np.argsort(num_speakers)
            num_speakers = num_speakers[sorting_indices]
            counts = counts[sorting_indices]

            ratios = np.cumsum(counts) / np.sum(counts)

            for k, ratio in zip(num_speakers, ratios):
                if k == 0:
                    print(f"   - {ratio:7.2%} of all chunks contain no speech at all.")
                elif k == 1:
                    print(f"   - {ratio:7.2%} contain 1 speaker or less")
                else:
                    print(f"   - {ratio:7.2%} contain {k} speakers or less")

            self.max_speakers_per_chunk = max(
                2,
                num_speakers[np.where(ratios > 0.97)[0][0]],
            )

            print(
                f"Setting `max_speakers_per_chunk` to {self.max_speakers_per_chunk}. "
                f"You can override this value (or avoid this estimation step) by passing `max_speakers_per_chunk={self.max_speakers_per_chunk}` to the task constructor."
            )

        if (
            self.max_speakers_per_frame is not None
            and self.max_speakers_per_frame > self.max_speakers_per_chunk
        ):
            raise ValueError(
                f"`max_speakers_per_frame` ({self.max_speakers_per_frame}) must be smaller "
                f"than `max_speakers_per_chunk` ({self.max_speakers_per_chunk})"
            )

        # now that we know about the number of speakers upper bound
        # we can set task specifications
        speaker_diarization = Specifications(
            duration=self.duration,
            resolution=Resolution.FRAME,
            problem=(
                Problem.MULTI_LABEL_CLASSIFICATION
                if self.max_speakers_per_frame is None
                else Problem.MONO_LABEL_CLASSIFICATION
            ),
            permutation_invariant=True,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
        )

        speaker_separation = Specifications(
            duration=self.duration,
            resolution=Resolution.FRAME,
            problem=Problem.MONO_LABEL_CLASSIFICATION,  # Doesn't matter
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
        )

        self.specifications = (speaker_diarization, speaker_separation)

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk

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
            - `y`: target as a SlidingWindowFeature instance where y.labels is
                   in meta.scope space.
            - `meta`:
                - `scope`: target scope (0: file, 1: database, 2: global)
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        # get label scope
        label_scope = Scopes[self.prepared_data["audio-metadata"][file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        #
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

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start : end + 1, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(y, self.model.receptive_field, labels=labels)

        metadata = self.prepared_data["audio-metadata"][file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

    def val_dataloader(self) -> DataLoader:
        """Validation data loader

        Returns
        -------
        DataLoader
            Validation data loader.
        """
        return DataLoader(
            ValDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=partial(self.collate_fn, stage="train"),
        )

    def val__iter__(self):
        """Iterate over validation samples

        Yields
        ------
        dict:
            X: (time, channel)
                Audio chunks.
            y: (frame, )
                Frame-level targets. Note that frame < time.
                `frame` is infered automagically from the
                example model output.
            ...
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.model)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.val__iter__helper(rng)

        else:
            # create a subchunk generator for each combination of "balance" keys
            subchunks = dict()
            for product in itertools.product(
                [self.metadata_unique_values[key] for key in balance]
            ):
                filters = {key: value for key, value in zip(balance, product)}
                subchunks[product] = self.val__iter__helper(rng, **filters)

        while True:
            # select one subchunk generator at random (with uniform probability)
            # so that it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(subchunks)]

            # generate random chunk
            yield next(chunks)

    def common__iter__helper(self, split, rng: random.Random, **filters):
        """Iterate over samples with optional domain filtering

        Mixtures are paired so that they have no speakers in common, come from the
        same file, and the combined number of speakers is no greater than
        max_speaker_per_chunk.

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key: value} dict), filter files so that
            only files such as file[key] == value are used for generating chunks.

        Yields
        ------
        chunk : dict
            Chunks.
        """
        # indices of files that matches domain filters
        split_files = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            split
        )
        for key, value in filters.items():
            split_files &= self.prepared_data["audio-metadata"][
                key
            ] == self.prepared_data["metadata"][key].index(value)
        file_ids = np.where(split_files)[0]

        # turn annotated duration into a probability distribution
        annotated_duration = self.prepared_data["audio-annotated"][file_ids]
        cum_prob_annotated_duration = np.cumsum(
            annotated_duration / np.sum(annotated_duration)
        )

        duration = self.duration

        num_chunks_per_file = getattr(self, "num_chunks_per_file", 1)

        while True:
            # select one file at random (with probability proportional to its annotated duration)
            file_id = file_ids[cum_prob_annotated_duration.searchsorted(rng.random())]
            annotations = self.prepared_data["annotations-segments"][
                np.where(
                    self.prepared_data["annotations-segments"]["file_id"] == file_id
                )[0]
            ]

            # generate `num_chunks_per_file` chunks from this file
            for _ in range(num_chunks_per_file):
                # find indices of annotated regions in this file
                annotated_region_indices = np.where(
                    self.prepared_data["annotations-regions"]["file_id"] == file_id
                )[0]

                # turn annotated regions duration into a probability distribution

                cum_prob_annotated_regions_duration = np.cumsum(
                    self.prepared_data["annotations-regions"]["duration"][
                        annotated_region_indices
                    ]
                    / np.sum(
                        self.prepared_data["annotations-regions"]["duration"][
                            annotated_region_indices
                        ]
                    )
                )

                # selected one annotated region at random (with probability proportional to its duration)
                annotated_region_index = annotated_region_indices[
                    cum_prob_annotated_regions_duration.searchsorted(rng.random())
                ]

                # select one chunk at random in this annotated region
                _, region_duration, start = self.prepared_data["annotations-regions"][
                    annotated_region_index
                ]
                start_time = rng.uniform(start, start + region_duration - duration)

                # find speakers that already appeared and all annotations that contain them
                chunk_annotations = annotations[
                    (annotations["start"] < start_time + duration)
                    & (annotations["end"] > start_time)
                ]
                previous_speaker_labels = list(
                    np.unique(chunk_annotations["file_label_idx"])
                )
                repeated_speaker_annotations = annotations[
                    np.isin(annotations["file_label_idx"], previous_speaker_labels)
                ]

                if repeated_speaker_annotations.size == 0:
                    # if previous chunk has 0 speakers then just sample from all annotated regions again
                    first_chunk = self.prepare_chunk(file_id, start_time, duration)

                    # selected one annotated region at random (with probability proportional to its duration)
                    annotated_region_index = np.random.choice(
                        annotated_region_indices, p=cum_prob_annotated_regions_duration
                    )

                    # select one chunk at random in this annotated region
                    _, region_duration, start = self.prepared_data[
                        "annotations-regions"
                    ][annotated_region_index]
                    start_time = rng.uniform(start, start + region_duration - duration)

                    second_chunk = self.prepare_chunk(file_id, start_time, duration)

                    labels = first_chunk["y"].labels + second_chunk["y"].labels

                    if len(labels) <= self.max_speakers_per_chunk:
                        yield first_chunk
                        yield second_chunk

                else:
                    # merge segments that contain repeated speakers
                    merged_repeated_segments = [
                        [
                            repeated_speaker_annotations["start"][0],
                            repeated_speaker_annotations["end"][0],
                        ]
                    ]
                    for _, start, end, _, _, _ in repeated_speaker_annotations:
                        previous = merged_repeated_segments[-1]
                        if start <= previous[1]:
                            previous[1] = max(previous[1], end)
                        else:
                            merged_repeated_segments.append([start, end])

                    # find segments that don't contain repeated speakers
                    segments_without_repeat = []
                    current_region_index = 0
                    previous_time = self.prepared_data["annotations-regions"]["start"][
                        annotated_region_indices[0]
                    ]
                    for segment in merged_repeated_segments:
                        if (
                            segment[0]
                            > self.prepared_data["annotations-regions"]["start"][
                                annotated_region_indices[current_region_index]
                            ]
                            + self.prepared_data["annotations-regions"]["duration"][
                                annotated_region_indices[current_region_index]
                            ]
                        ):
                            current_region_index += 1
                            previous_time = self.prepared_data["annotations-regions"][
                                "start"
                            ][annotated_region_indices[current_region_index]]

                        if segment[0] - previous_time > duration:
                            segments_without_repeat.append(
                                (previous_time, segment[0], segment[0] - previous_time)
                            )
                        previous_time = segment[1]

                    dtype = [("start", "f"), ("end", "f"), ("duration", "f")]
                    segments_without_repeat = np.array(
                        segments_without_repeat, dtype=dtype
                    )

                    if np.sum(segments_without_repeat["duration"]) != 0:
                        # only yield chunks if it is possible to choose the second chunk so that yielded chunks are always paired
                        first_chunk = self.prepare_chunk(file_id, start_time, duration)

                        prob_segments_duration = segments_without_repeat[
                            "duration"
                        ] / np.sum(segments_without_repeat["duration"])
                        segment = np.random.choice(
                            segments_without_repeat, p=prob_segments_duration
                        )

                        start, end, _ = segment
                        new_start_time = rng.uniform(start, end - duration)
                        second_chunk = self.prepare_chunk(
                            file_id, new_start_time, duration
                        )

                        labels = first_chunk["y"].labels + second_chunk["y"].labels
                        if len(labels) <= self.max_speakers_per_chunk:
                            yield first_chunk
                            yield second_chunk

    def val__iter__helper(self, rng: random.Random, **filters):
        """Iterate over validation samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key: value} dict), filter validation files so that
            only files such as file[key] == value are used for generating chunks.

        Yields
        ------
        chunk : dict
            validation chunks.
        """

        return self.common__iter__helper("development", rng, **filters)

    def train__iter__helper(self, rng: random.Random, **filters):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key: value} dict), filter training files so that
            only files such as file[key] == value are used for generating chunks.

        Yields
        ------
        chunk : dict
            Training chunks.
        """

        return self.common__iter__helper("train", rng, **filters)

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
        collated_y = self.collate_y(batch)

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

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
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

        seg_loss = binary_cross_entropy(
            permutated_prediction, target.float(), weight=weight
        )

        return seg_loss

    def create_mixtures_of_mixtures(self, mix1, mix2, target1, target2):
        """
        Creates mixtures of mixtures and corresponding diarization targets.
        Keeps track of how many speakers came from each mixture in order to
        reconstruct the original mixtures.

        Parameters
        ----------
        mix1 : torch.Tensor
            First mixture.
        mix2 : torch.Tensor
            Second mixture.
        target1 : torch.Tensor
            First mixture diarization targets.
        target2 : torch.Tensor
            Second mixture diarization targets.

        Returns
        -------
        mom : torch.Tensor
            Mixtures of mixtures.
        targets : torch.Tensor
            Diarization targets for mixtures of mixtures.
        num_active_speakers_mix1 : torch.Tensor
            Number of active speakers in the first mixture.
        num_active_speakers_mix2 : torch.Tensor
            Number of active speakers in the second mixture.
        """
        batch_size = mix1.shape[0]
        mom = mix1 + mix2
        num_active_speakers_mix1 = (target1.sum(dim=1) != 0).sum(dim=1)
        num_active_speakers_mix2 = (target2.sum(dim=1) != 0).sum(dim=1)
        targets = []
        for i in range(batch_size):
            target = torch.cat(
                (
                    target1[i][:, target1[i].sum(dim=0) != 0],
                    target2[i][:, target2[i].sum(dim=0) != 0],
                ),
                dim=1,
            )
            padding_dim = (
                target1.shape[2]
                - num_active_speakers_mix1[i]
                - num_active_speakers_mix2[i]
            )
            padding_tensor = torch.zeros(
                (target1.shape[1], padding_dim), device=target.device
            )
            target = torch.cat((target, padding_tensor), dim=1)
            targets.append(target)
        targets = torch.stack(targets)

        return mom, targets, num_active_speakers_mix1, num_active_speakers_mix2

    def common_step(self, batch):
        """Common step for training and validation

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.

        Returns
        -------
        seg_loss : torch.Tensor
            Segmentation loss.
        separation_loss : torch.Tensor
            Separation loss.
        diarization : torch.Tensor
            Diarization predictions.
        permutated_diarization : torch.Tensor
            Permutated diarization predictions that minizimise seg_loss.
        target : torch.Tensor
            Diarization target.
        """

        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # forward pass
        bsz = waveform.shape[0]

        # MoMs can't be created for batch size < 2
        if bsz < 2:
            return None
        # if bsz not even, then leave out last sample
        if bsz % 2 != 0:
            waveform = waveform[:-1]

        mix1 = waveform[0::2].squeeze(1)
        mix2 = waveform[1::2].squeeze(1)

        (
            mom,
            mom_target,
            _,
            _,
        ) = self.create_mixtures_of_mixtures(mix1, mix2, target[0::2], target[1::2])
        target = torch.cat((target[0::2], target[1::2], mom_target), dim=0)

        diarization, sources = self.model(torch.cat((mix1, mix2, mom), dim=0))
        mom_sources = sources[bsz:]

        batch_size, num_frames, _ = diarization.shape
        # (batch_size, num_frames, num_classes)

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        permutated_diarization, _ = permutate(target, diarization)

        seg_loss = self.segmentation_loss(permutated_diarization, target, weight=weight)

        separation_loss = self.mixit_loss(
            mom_sources.transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1)
        ).mean()

        return (
            seg_loss,
            separation_loss,
            diarization,
            permutated_diarization,
            target,
        )

    def training_step(self, batch, batch_idx: int):
        """Compute PixIT loss for training

        Parameters
        ----------
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss}
        """
        # finetuning wavlm with a smaller learning rate requires two optimizers
        # and manual gradient stepping
        if self.finetune_wavlm:
            wavlm_opt, rest_opt = self.model.optimizers()
            wavlm_opt.zero_grad()
            rest_opt.zero_grad()

        (
            seg_loss,
            separation_loss,
            diarization,
            permutated_diarization,
            target,
        ) = self.common_step(batch)
        self.model.log(
            "loss/train/separation",
            separation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.model.log(
            "loss/train/segmentation",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        loss = (
            1 - self.separation_loss_weight
        ) * seg_loss + self.separation_loss_weight * separation_loss

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

        if self.finetune_wavlm:
            self.model.manual_backward(loss)
            self.model.clip_gradients(
                wavlm_opt,
                gradient_clip_val=self.model.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            self.model.clip_gradients(
                rest_opt,
                gradient_clip_val=self.model.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            wavlm_opt.step()
            rest_opt.step()

        return {"loss": loss}

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components"""

        return {
            "DiarizationErrorRate": OptimalDiarizationErrorRate(),
            "DiarizationErrorRate/Threshold": OptimalDiarizationErrorRateThreshold(),
            "DiarizationErrorRate/Confusion": OptimalSpeakerConfusionRate(),
            "DiarizationErrorRate/Miss": OptimalMissedDetectionRate(),
            "DiarizationErrorRate/FalseAlarm": OptimalFalseAlarmRate(),
        }

    # TODO: no need to compute gradient in this method
    def validation_step(self, batch, batch_idx: int):
        """Compute validation loss and metric

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        (
            seg_loss,
            separation_loss,
            diarization,
            permutated_diarization,
            target,
        ) = self.common_step(batch)

        self.model.log(
            "loss/val/separation",
            separation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.model.log(
            "loss/val/segmentation",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        loss = (
            1 - self.separation_loss_weight
        ) * seg_loss + self.separation_loss_weight * separation_loss

        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.model.validation_metric(
            torch.transpose(diarization, 1, 2),
            torch.transpose(target, 1, 2),
        )

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # log first batch visualization every 2^n epochs.
        if (
            self.model.current_epoch == 0
            or math.log2(self.model.current_epoch) % 1 > 0
            or batch_idx > 0
        ):
            return

        # visualize first 9 validation samples of first batch in Tensorboard/MLflow

        y = target.float().cpu().numpy()
        y_pred = permutated_diarization.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.nan
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):
            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot target
            ax_ref = axes[row_idx * 2 + 0, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot predictions
            ax_hyp = axes[row_idx * 2 + 1, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

        plt.tight_layout()

        for logger in self.model.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("samples", fig, self.model.current_epoch)
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"samples_epoch{self.model.current_epoch}.png",
                )

        plt.close(fig)
