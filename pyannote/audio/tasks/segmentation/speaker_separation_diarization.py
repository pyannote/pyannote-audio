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

import math
import warnings
import random
from collections import Counter
from typing import Dict, Literal, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from rich.progress import track
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.utils.loss import binary_cross_entropy, mse_loss, nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset
from asteroid.losses import multisrc_neg_sisdr
from torch.utils.data._utils.collate import default_collate

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)

from itertools import combinations
from torch import nn

class ModifiedMixITLossWrapper(nn.Module):
    r"""Mixture invariant loss wrapper modifed to force alignment between separation and diarization.

    Args:
        loss_func: function with signature (est_targets, targets, **kwargs).
        generalized (bool): Determines how MixIT is applied. If False ,
            apply MixIT for any number of mixtures as soon as they contain
            the same number of sources (:meth:`~MixITLossWrapper.best_part_mixit`.)
            If True (default), apply MixIT for two mixtures, but those mixtures do not
            necessarly have to contain the same number of sources.
            See :meth:`~MixITLossWrapper.best_part_mixit_generalized`.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

    For each of these modes, the best partition and reordering will be
    automatically computed.

    Examples:
        >>> import torch
        >>> from asteroid.losses import multisrc_mse
        >>> mixtures = torch.randn(10, 2, 16000)
        >>> est_sources = torch.randn(10, 4, 16000)
        >>> # Compute MixIT loss based on pairwise losses
        >>> loss_func = MixITLossWrapper(multisrc_mse)
        >>> loss_val = loss_func(est_sources, mixtures)

    References
        [1] Scott Wisdom et al. "Unsupervised sound separation using
        mixtures of mixtures." arXiv:2006.12701 (2020)
    """

    def __init__(self, loss_func, generalized=True, reduction="mean"):
        super().__init__()
        self.loss_func = loss_func
        self.generalized = generalized
        self.reduction = reduction

    def forward(self, est_targets, targets, part_from_mix1, part_from_mix2, return_est=False, **kwargs):
        r"""Find the best partition and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, *)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets
            return_est: Boolean. Whether to return the estimated mixtures
                estimates (To compute metrics or to save example).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best partition loss for each batch sample, average over
              the batch. torch.Tensor(loss_value)
            - The estimated mixtures (estimated sources summed according to the partition)
              if return_est is True. torch.Tensor of shape :math:`(batch, nmix, ...)`.
        """
        # Check input dimensions
        assert est_targets.shape[0] == targets.shape[0]
        assert est_targets.shape[2] == targets.shape[2]

        # if not self.generalized:
        #     min_loss, min_loss_idx, parts = self.best_part_mixit(
        #         self.loss_func, est_targets, targets, **kwargs
        #     )
        # else:
        #     min_loss, min_loss_idx, parts = self.best_part_mixit_generalized(
        #         self.loss_func, est_targets, targets, **kwargs
        #     )
        est_mixes = []
        for i in range(est_targets.shape[0]):
            # sum the sources according to the given partition
            est_mix1 = est_targets[i, part_from_mix1[i], :].sum(0)
            est_mix2 = est_targets[i, part_from_mix2[i], :].sum(0)
            # get loss for the given partition
            
            est_mixes.append(torch.stack((est_mix1, est_mix2)))
        est_mixes = torch.stack(est_mixes)
        loss_partition = self.loss_func(est_mixes, targets, **kwargs)
        if loss_partition.ndim != 1:
            raise ValueError("Loss function return value should be of size (batch,).")

        # Apply any reductions over the batch axis
        returned_loss = loss_partition.mean() if self.reduction == "mean" else loss_partition
        if not return_est:
            return returned_loss

        # Order and sum on the best partition to get the estimated mixtures
        # reordered = self.reorder_source(est_targets, targets, min_loss_idx, parts)
        return returned_loss, est_mixes

class JointSpeakerSeparationAndDiarization(SegmentationTaskMixin, Task):
    """Speaker diarization

    Parameters
    ----------
    protocol : SpeakerDiarizationProtocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
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
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "database" will make sure that each database
        will be equally represented in the training samples.
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
    vad_loss : {"bce", "mse"}, optional
        Add voice activity detection loss.
        Cannot be used in conjunction with `max_speakers_per_frame`.
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    mixit_loss_weight : float, optional
        Factor that speaker separation loss is scaled by when calculating total loss.

    References
    ----------
    Hervé Bredin and Antoine Laurent
    "End-To-End Speaker Segmentation for Overlap-Aware Resegmentation."
    Proc. Interspeech 2021

    Zhihao Du, Shiliang Zhang, Siqi Zheng, and Zhijie Yan
    "Speaker Embedding-aware Neural Diarization: an Efficient Framework for Overlapping
    Speech Diarization in Meeting Scenarios"
    https://arxiv.org/abs/2203.09767

    """

    def __init__(
        self,
        protocol: SpeakerDiarizationProtocol,
        duration: float = 2.0,
        max_speakers_per_chunk: int = None,
        max_speakers_per_frame: int = None,
        weigh_by_cardinality: bool = False,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        vad_loss: Literal["bce", "mse"] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        max_num_speakers: int = None,  # deprecated in favor of `max_speakers_per_chunk``
        loss: Literal["bce", "mse"] = None,  # deprecated
        mixit_loss_weight: float = 0.2,
    ):
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
            if max_speakers_per_frame < 1:
                raise ValueError(
                    f"`max_speakers_per_frame` must be 1 or more (you used {max_speakers_per_frame})."
                )
            if vad_loss is not None:
                raise ValueError(
                    "`vad_loss` cannot be used jointly with `max_speakers_per_frame`"
                )

        if batch_size % 3 != 0:
            raise ValueError("`batch_size` must be divisible by 3 for mixtures of mixtures training")  

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.weigh_by_cardinality = weigh_by_cardinality
        self.balance = balance
        self.weight = weight
        self.vad_loss = vad_loss
        self.separation_loss = ModifiedMixITLossWrapper(multisrc_neg_sisdr, generalized=True)
        self.mixit_loss_weight = mixit_loss_weight

    def setup(self):
        super().setup()

        # estimate maximum number of speakers per chunk when not provided
        if self.max_speakers_per_chunk is None:
            training = self.metadata["subset"] == Subsets.index("train")

            num_unique_speakers = []
            progress_description = f"Estimating maximum number of speakers per {self.duration:g}s chunk in the training set"
            for file_id in track(
                np.where(training)[0], description=progress_description
            ):
                annotations = self.annotations[
                    np.where(self.annotations["file_id"] == file_id)[0]
                ]
                annotated_regions = self.annotated_regions[
                    np.where(self.annotated_regions["file_id"] == file_id)[0]
                ]
                for region in annotated_regions:
                    # find annotations within current region
                    region_start = region["start"]
                    region_end = region["end"]
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
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            permutation_invariant=True,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
        )

        speaker_separation = Specifications(
            duration=self.duration,
            resolution=Resolution.FRAME,
            problem=Problem.MONO_LABEL_CLASSIFICATION, # Doesn't matter
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
        )

        self.specifications = (speaker_diarization, speaker_separation)

    def setup_loss_func(self):
        if self.specifications[0].powerset:
            self.model.powerset = Powerset(
                len(self.specifications[0].classes),
                self.specifications[0].powerset_max_classes,
            )

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
        label_scope = Scopes[self.metadata[file_id]["scope"]]
        label_scope_key = f"{label_scope}_label_idx"

        #
        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # use model introspection to predict how many frames it will output
        # TODO: this should be cached
        num_samples = sample["X"].shape[1]
        #resolution_samples = self.model.example_output[0].frames.step * self.model.example_output[0].num_frames / num_samples

        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output and input resolutions
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        start_idx = np.floor(start / self.model.example_output[0].frames.step).astype(int)
        start_idx_samples = np.floor(start * 16000).astype(int)
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / self.model.example_output[0].frames.step).astype(int)
        end_idx_samples = np.floor(end * 16000).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        y = np.zeros((self.model.example_output[0].num_frames, num_labels), dtype=np.uint8)
        sample_level_labels = np.zeros((num_samples, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            y[start:end, mapped_label] = 1

        sample["y"] = SlidingWindowFeature(
            y, self.model.example_output[0].frames, labels=labels
        )
        
        for start, end, label in zip(
            start_idx_samples, end_idx_samples, chunk_annotations[label_scope_key]
        ):
            mapped_label = mapping[label]
            sample_level_labels[start:end, mapped_label] = 1

        # only frames with a single label should be used for mixit training
        sample["X_separation_mask"] = torch.from_numpy(sample_level_labels.sum(axis=1) == 1)
        metadata = self.metadata[file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample

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

        # indices of training files that matches domain filters
        training = self.metadata["subset"] == Subsets.index("train")
        for key, value in filters.items():
            training &= self.metadata[key] == value
        file_ids = np.where(training)[0]

        # turn annotated duration into a probability distribution
        annotated_duration = self.annotated_duration[file_ids]
        prob_annotated_duration = annotated_duration / np.sum(annotated_duration)

        duration = self.duration

        num_chunks_per_file = getattr(self, "num_chunks_per_file", 1)

        while True:
            # select one file at random (with probability proportional to its annotated duration)
            file_id = np.random.choice(file_ids, p=prob_annotated_duration)
            annotations = self.annotations[np.where(self.annotations["file_id"] == file_id)[0]]

            # generate `num_chunks_per_file` chunks from this file
            for _ in range(num_chunks_per_file):
                # find indices of annotated regions in this file
                annotated_region_indices = np.where(
                    self.annotated_regions["file_id"] == file_id
                )[0]

                # turn annotated regions duration into a probability distribution
                prob_annotated_regions_duration = self.annotated_regions["duration"][
                    annotated_region_indices
                ] / np.sum(self.annotated_regions["duration"][annotated_region_indices])

                # selected one annotated region at random (with probability proportional to its duration)
                annotated_region_index = np.random.choice(
                    annotated_region_indices, p=prob_annotated_regions_duration
                )

                # select one chunk at random in this annotated region
                _, _, start, end = self.annotated_regions[annotated_region_index]
                start_time = rng.uniform(start, end - duration)

                # find speakers that already appeared and all annotations that contain them
                chunk_annotations = annotations[
                    (annotations["start"] < start_time+duration) & (annotations["end"] > start_time)
                ]
                previous_speaker_labels = list(np.unique(chunk_annotations["file_label_idx"]))
                repeated_speaker_annotations = annotations[np.isin(annotations["file_label_idx"], previous_speaker_labels)]
                
                if repeated_speaker_annotations.size == 0:
                    # if previous chunk has 0 speakers then just sample from all annotated regions again
                    first_chunk = self.prepare_chunk(file_id, start_time, duration)
                    first_chunk["meta"]["mixture_type"]="first_mixture"
                    # in order to align separation and diarization branches we need to know which mixtures do speakers/sources originate from
                    first_chunk["meta"]["sources_from_first_mixture"] = len(first_chunk["y"].labels)
                    first_chunk["meta"]["sources_from_second_mixture"] = 0
                    # yield first_chunk

                    # selected one annotated region at random (with probability proportional to its duration)
                    annotated_region_index = np.random.choice(
                        annotated_region_indices, p=prob_annotated_regions_duration
                    )

                    # select one chunk at random in this annotated region
                    _, _, start, end = self.annotated_regions[annotated_region_index]
                    start_time = rng.uniform(start, end - duration)

                    second_chunk = self.prepare_chunk(file_id, start_time, duration)
                    second_chunk["meta"]["mixture_type"]="second_mixture"
                    second_chunk["meta"]["sources_from_first_mixture"] = 0
                    second_chunk["meta"]["sources_from_second_mixture"] = len(second_chunk["y"].labels)
                    # yield second_chunk

                    # add previous two chunks to get a third one
                    third_chunk = dict()
                    third_chunk["X"] = first_chunk["X"] + second_chunk["X"]
                    third_chunk["meta"] = first_chunk["meta"].copy()
                    y = np.concatenate((first_chunk["y"].data, second_chunk["y"].data), axis=1)
                    frames = first_chunk["y"].sliding_window
                    labels = first_chunk["y"].labels + second_chunk["y"].labels
                    third_chunk["y"] = SlidingWindowFeature(y, frames, labels=labels)
                    third_chunk["meta"]["mixture_type"]="mom"
                    third_chunk["meta"]["sources_from_first_mixture"] = len(first_chunk["y"].labels)
                    third_chunk["meta"]["sources_from_second_mixture"] = len(second_chunk["y"].labels)

                    # the whole mom should be used in the separation branch training
                    third_chunk["X_separation_mask"] = torch.ones_like(first_chunk["X_separation_mask"])

                    if len(labels) < 4:
                        yield first_chunk
                        yield second_chunk
                        yield third_chunk
                    
                else:
                    # merge segments that contain repeated speakers
                    merged_repeated_segments = [[repeated_speaker_annotations["start"][0],repeated_speaker_annotations["end"][0]]]
                    for _, start, end, _, _, _ in repeated_speaker_annotations:
                        previous = merged_repeated_segments[-1]
                        if start <= previous[1]:
                            previous[1] = max(previous[1], end)
                        else:
                            merged_repeated_segments.append([start, end])
                    
                    # find segments that don't contain repeated speakers
                    segments_without_repeat = []
                    current_region_index = 0
                    previous_time = self.annotated_regions["start"][annotated_region_indices[0]]
                    for segment in merged_repeated_segments:
                        if segment[0] > self.annotated_regions["end"][annotated_region_indices[current_region_index]]:
                            current_region_index+=1
                            previous_time = self.annotated_regions["start"][annotated_region_indices[current_region_index]]
                        
                        if segment[0] - previous_time > duration:
                            segments_without_repeat.append((previous_time, segment[0], segment[0] - previous_time))
                        previous_time = segment[1]
                    
                    dtype = [("start", "f"), ("end", "f"),("duration", "f")]
                    segments_without_repeat = np.array(segments_without_repeat,dtype=dtype)

                    if np.sum(segments_without_repeat["duration"]) != 0:

                        # only yield chunks if it is possible to choose the second chunk so that yielded chunks are always paired

                        first_chunk = self.prepare_chunk(file_id, start_time, duration)
                        first_chunk["meta"]["mixture_type"]="first_mixture"
                        first_chunk["meta"]["sources_from_first_mixture"] = len(first_chunk["y"].labels)
                        first_chunk["meta"]["sources_from_second_mixture"] = 0
                        #yield first_chunk

                        prob_segments_duration = segments_without_repeat["duration"] / np.sum(segments_without_repeat["duration"])
                        segment = np.random.choice(
                            segments_without_repeat, p=prob_segments_duration
                        )

                        start, end, _ = segment
                        new_start_time = rng.uniform(start, end - duration)
                        second_chunk = self.prepare_chunk(file_id, new_start_time, duration)
                        second_chunk["meta"]["mixture_type"]="second_mixture"
                        second_chunk["meta"]["sources_from_first_mixture"] = 0
                        second_chunk["meta"]["sources_from_second_mixture"] = len(second_chunk["y"].labels)
                        #yield second_chunk

                        #add previous two chunks to get a third one
                        third_chunk = dict()
                        third_chunk["X"] = first_chunk["X"] + second_chunk["X"]
                        third_chunk["meta"] = first_chunk["meta"].copy()
                        y = np.concatenate((first_chunk["y"].data, second_chunk["y"].data), axis=1)
                        frames = first_chunk["y"].sliding_window
                        labels = first_chunk["y"].labels + second_chunk["y"].labels
                        third_chunk["y"] = SlidingWindowFeature(y, frames, labels=labels)
                        third_chunk["meta"]["mixture_type"]="mom"

                        # the whole mom should be used in the separation branch training
                        third_chunk["X_separation_mask"] = torch.ones_like(first_chunk["X_separation_mask"])
                        third_chunk["meta"]["sources_from_first_mixture"] = len(first_chunk["y"].labels)
                        third_chunk["meta"]["sources_from_second_mixture"] = len(second_chunk["y"].labels)
                        #third_chunk["sources_from_first_mixture"] = len(first_chunk["y"].labels)
                        if len(labels) < 4:
                            yield first_chunk
                            yield second_chunk
                            yield third_chunk

    def collate_X_separation_mask(self, batch) -> torch.Tensor:
        return default_collate([b["X_separation_mask"] for b in batch])

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

        collated_X_separation_mask = self.collate_X_separation_mask(batch)

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
            "X_separation_mask" : collated_X_separation_mask
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

        if self.specifications[0].powerset:
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
        else:
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
            )

        return seg_loss

    def voice_activity_detection_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Voice activity detection loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        vad_loss : torch.Tensor
            Voice activity detection loss.
        """

        vad_prediction, _ = torch.max(permutated_prediction, dim=2, keepdim=True)
        # (batch_size, num_frames, 1)

        vad_target, _ = torch.max(target.float(), dim=2, keepdim=False)
        # (batch_size, num_frames)

        if self.vad_loss == "bce":
            loss = binary_cross_entropy(vad_prediction, vad_target, weight=weight)

        elif self.vad_loss == "mse":
            loss = mse_loss(vad_prediction, vad_target, weight=weight)

        return loss

    def training_step(self, batch, batch_idx: int):
        """Compute permutation-invariant segmentation loss

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

        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # drop samples that contain too many speakers
        num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
        keep: torch.Tensor = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        waveform = waveform[keep]

        # corner case
        if not keep.any():
            return None

        # forward pass
        bsz = waveform.shape[0]
        num_samples = waveform.shape[2]
        mix1 = waveform[0::3].squeeze(1)
        mix2 = waveform[1::3].squeeze(1)
        # extract parts with only one speaker from original mixtures
        mix1_masks = batch["X_separation_mask"][0::3]
        mix2_masks = batch["X_separation_mask"][1::3]
        mix1_masked = mix1 * mix1_masks
        mix2_masked = mix2 * mix2_masks

        moms = mix1 + mix2
        
        _, predicted_sources_mom = self.model(moms)
        _, predicted_sources_mix1 = self.model(mix1)
        _, predicted_sources_mix2 = self.model(mix2)

        # don't use moms with more than max_speakers_per_chunk speakers for training speaker diarization
        num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
        num_speakers[2::3] = num_speakers[::3] + num_speakers[1::3]
        keep: torch.Tensor = num_speakers <= self.max_speakers_per_chunk
        target = target[keep]
        waveform = waveform[keep]
        prediction, _ = self.model(waveform)

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

        if self.specifications[0].powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            permutated_target, permutations = permutate(multilabel, target)
            permutated_target_powerset = self.model.powerset.to_powerset(
                permutated_target.float()
            )
            seg_loss = self.segmentation_loss(
                prediction, permutated_target_powerset, weight=weight
            )

        else:
            permutated_prediction, permutations = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )
        # to find which predicted sources correspond to which mixtures, we need to invert the permutations
        permutations_inverse = torch.argsort(torch.tensor(permutations))
        predicted_sources_idx_mix1 = [[permutations_inverse[i][j] for j in range(batch["meta"]["sources_from_first_mixture"][i])] for i in range(batch_size)]
        predicted_sources_idx_mix2 = [[permutations_inverse[i][j] for j in range(batch["meta"]["sources_from_first_mixture"][i],batch["meta"]["sources_from_second_mixture"][i])] for i in range(batch_size)]
        # contributions from original mixtures is weighed by the proportion of remaining frames
        mixit_loss = self.separation_loss(
            predicted_sources_mom.transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1), predicted_sources_idx_mix1[2::3], predicted_sources_idx_mix2[2::3]
        ) + self.separation_loss(
            predicted_sources_mix1.transpose(1, 2), torch.stack((mix1_masked, torch.zeros_like(mix1))).transpose(0, 1), predicted_sources_idx_mix1[0::3], predicted_sources_idx_mix2[0::3]
        ) * mix1_masks.sum() / num_samples / bsz * 3 + self.separation_loss(
            predicted_sources_mix2.transpose(1, 2), torch.stack((mix2_masked, torch.zeros_like(mix2))).transpose(0, 1), predicted_sources_idx_mix1[1::3], predicted_sources_idx_mix2[1::3]
        ) * mix2_masks.sum() / num_samples / bsz * 3

        self.model.log(
            "loss/train/separation",
            mixit_loss,
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

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            # TODO: vad_loss probably does not make sense in powerset mode
            # because first class (empty set of labels) does exactly this...
            if self.specifications[0].powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                "loss/train/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = (1 - self.mixit_loss_weight) * (seg_loss + vad_loss) + self.mixit_loss_weight * mixit_loss

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

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components"""

        if self.specifications[0].powerset:
            return {
                "DiarizationErrorRate": DiarizationErrorRate(0.5),
                "DiarizationErrorRate/Confusion": SpeakerConfusionRate(0.5),
                "DiarizationErrorRate/Miss": MissedDetectionRate(0.5),
                "DiarizationErrorRate/FalseAlarm": FalseAlarmRate(0.5),
            }

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

        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # TODO: should we handle validation samples with too many speakers
        # waveform = waveform[keep]
        # target = target[keep]

        bsz = waveform.shape[0]
        mix1 = waveform[bsz // 2 : 2 * (bsz // 2)].squeeze(1)
        mix2 = waveform[: bsz // 2].squeeze(1)
        moms = mix1 + mix2

        # forward pass
        prediction, _ = self.model(waveform)
        _, predicted_sources_mom = self.model(moms)
        batch_size, num_frames, _ = prediction.shape

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

        if self.specifications[0].powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            permutated_target, _ = permutate(multilabel, target)

            # FIXME: handle case where target have too many speakers?
            # since we don't need
            permutated_target_powerset = self.model.powerset.to_powerset(
                permutated_target.float()
            )
            seg_loss = self.segmentation_loss(
                prediction, permutated_target_powerset, weight=weight
            )

        else:
            permutated_prediction, _ = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )
        # forced alignment mixit can't be implemented for validation because since data loading is different
        mixit_loss = 0
        # mixit_loss = self.separation_loss(
        #     predicted_sources_mom.transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1)
        # )

        self.model.log(
            "loss/val/separation",
            mixit_loss,
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

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            # TODO: vad_loss probably does not make sense in powerset mode
            # because first class (empty set of labels) does exactly this...
            if self.specifications[0].powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                "loss/val/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = (1 - self.mixit_loss_weight) * (seg_loss + vad_loss) + self.mixit_loss_weight * mixit_loss

        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.specifications[0].powerset:
            self.model.validation_metric(
                torch.transpose(
                    multilabel[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
            )
        else:
            self.model.validation_metric(
                torch.transpose(
                    prediction[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
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

        if self.specifications[0].powerset:
            y = permutated_target.float().cpu().numpy()
            y_pred = multilabel.cpu().numpy()
        else:
            y = target.float().cpu().numpy()
            y_pred = permutated_prediction.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.NaN
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
            ax_hyp.axvspan(0, warm_up_left, color="k", alpha=0.5, lw=0)
            ax_hyp.axvspan(
                num_frames - warm_up_right, num_frames, color="k", alpha=0.5, lw=0
            )
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


def main(protocol: str, subset: str = "test", model: str = "pyannote/segmentation"):
    """Evaluate a segmentation model"""

    from pyannote.database import FileFinder, get_protocol
    from rich.progress import Progress

    from pyannote.audio import Inference
    from pyannote.audio.pipelines.utils import get_devices
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.utils.signal import binarize

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})
    files = list(getattr(protocol, subset)())

    with Progress() as progress:
        main_task = progress.add_task(protocol.name, total=len(files))
        file_task = progress.add_task("Processing", total=1.0)

        def progress_hook(completed: int = None, total: int = None):
            progress.update(file_task, completed=completed / total)

        inference = Inference(model, device=device)

        for file in files:
            progress.update(file_task, description=file["uri"])
            reference = file["annotation"]
            hypothesis = binarize(inference(file, hook=progress_hook))
            uem = file["annotated"]
            _ = metric(reference, hypothesis, uem=uem)
            progress.advance(main_task)

    _ = metric.report(display=True)


if __name__ == "__main__":
    import typer

    typer.run(main)
