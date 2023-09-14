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
import lightning.pytorch as pl

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
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr, PITLossWrapper, pairwise_neg_sisdr
from torch.utils.data._utils.collate import default_collate

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)

from itertools import combinations
from torch import nn
from pytorch_lightning.callbacks import Callback

class CountingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        "reset counters"
        if pl_module.task.log_alignment_accuracy and pl_module.task.force_alignment:
            pl_module.task.num_correct = 0
            pl_module.task.num_total = 0
            pl_module.task.num_correct30 = 0
            pl_module.task.num_correct21 = 0
            pl_module.task.num_correct20 = 0
            pl_module.task.num_correct11 = 0
            pl_module.task.num_correct10 = 0
            pl_module.task.num_total30 = 0
            pl_module.task.num_total21 = 0
            pl_module.task.num_total20 = 0
            pl_module.task.num_total11 = 0
            pl_module.task.num_total10 = 0

class CustomMixITLossWrapper(nn.Module):
    r"""Custom mixture invariant loss wrapper that returns the best partition
    so that it can be checked against the partition determined by forced
    alignment.

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

    def forward(self, est_targets, targets, return_est=False, **kwargs):
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

        if not self.generalized:
            min_loss, min_loss_idx, parts = self.best_part_mixit(
                self.loss_func, est_targets, targets, **kwargs
            )
        else:
            min_loss, min_loss_idx, parts = self.best_part_mixit_generalized(
                self.loss_func, est_targets, targets, **kwargs
            )

        # Apply any reductions over the batch axis
        returned_loss = min_loss.mean() if self.reduction == "mean" else min_loss
        if not return_est:
            return returned_loss, [parts[i] for i in min_loss_idx]

        # Order and sum on the best partition to get the estimated mixtures
        reordered = self.reorder_source(est_targets, targets, min_loss_idx, parts)
        return returned_loss, reordered

    @staticmethod
    def best_part_mixit(loss_func, est_targets, targets, **kwargs):
        r"""Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid for any number of
        mixtures as soon as they contain the same number of sources.

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indices of the best partition.

            - :class:`list`:
              list of the possible partitions of the sources.

        """
        nmix = targets.shape[1]
        nsrc = est_targets.shape[1]
        if nsrc % nmix != 0:
            raise ValueError("The mixtures are assumed to contain the same number of sources")
        nsrcmix = nsrc // nmix

        # Generate all unique partitions of size k from a list lst of
        # length n, where l = n // k is the number of parts. The total
        # number of such partitions is: NPK(n,k) = n! / ((k!)^l * l!)
        # Algorithm recursively distributes items over parts
        def parts_mixit(lst, k, l):
            if l == 0:
                yield []
            else:
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    for r in parts_mixit(rest, k, l - 1):
                        yield [list(c), *r]

        # Generate all the possible partitions
        parts = list(parts_mixit(range(nsrc), nsrcmix, nmix))
        # Compute the loss corresponding to each partition
        loss_set = CustomMixITLossWrapper.loss_set_from_parts(
            loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def best_part_mixit_generalized(loss_func, est_targets, targets, **kwargs):
        r"""Find best partition of the estimated sources that gives the minimum
        loss for the MixIT training paradigm in [1]. Valid only for two mixtures,
        but those mixtures do not necessarly have to contain the same number of
        sources e.g the case where one mixture is silent is allowed..

        Args:
            loss_func: function with signature ``(est_targets, targets, **kwargs)``
                The loss function to get batch losses from.
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets (mixtures).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - :class:`torch.Tensor`:
              The loss corresponding to the best permutation of size (batch,).

            - :class:`torch.LongTensor`:
              The indexes of the best permutations.

            - :class:`list`:
              list of the possible partitions of the sources.
        """
        nmix = targets.shape[1]  # number of mixtures
        nsrc = est_targets.shape[1]  # number of estimated sources
        if nmix != 2:
            raise ValueError("Works only with two mixtures")

        # Generate all unique partitions of any size from a list lst of
        # length n. Algorithm recursively distributes items over parts
        def parts_mixit_gen(lst):
            partitions = []
            for k in range(len(lst) + 1):
                for c in combinations(lst, k):
                    rest = [x for x in lst if x not in c]
                    partitions.append([list(c), rest])
            return partitions

        # Generate all the possible partitions
        parts = parts_mixit_gen(range(nsrc))
        # Compute the loss corresponding to each partition
        loss_set = CustomMixITLossWrapper.loss_set_from_parts(
            loss_func, est_targets=est_targets, targets=targets, parts=parts, **kwargs
        )
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_indexes = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_indexes, parts

    @staticmethod
    def loss_set_from_parts(loss_func, est_targets, targets, parts, **kwargs):
        """Common loop between both best_part_mixit"""
        loss_set = []
        for partition in parts:
            # sum the sources according to the given partition
            est_mixes = torch.stack([est_targets[:, idx, :].sum(1) for idx in partition], dim=1)
            # get loss for the given partition
            loss_partition = loss_func(est_mixes, targets, **kwargs)
            if loss_partition.ndim != 1:
                raise ValueError("Loss function return value should be of size (batch,).")
            loss_set.append(loss_partition[:, None])
        loss_set = torch.cat(loss_set, dim=1)
        return loss_set

    @staticmethod
    def reorder_source(est_targets, targets, min_loss_idx, parts):
        """Reorder sources according to the best partition.

        Args:
            est_targets: torch.Tensor. Expected shape :math:`(batch, nsrc, ...)`.
                The batch of target estimates.
            targets: torch.Tensor. Expected shape :math:`(batch, nmix, ...)`.
                The batch of training targets.
            min_loss_idx: torch.LongTensor. The indexes of the best permutations.
            parts: list of the possible partitions of the sources.

        Returns:
            :class:`torch.Tensor`: Reordered sources of shape :math:`(batch, nmix, time)`.

        """
        # For each batch there is a different min_loss_idx
        ordered = torch.zeros_like(targets)
        for b, idx in enumerate(min_loss_idx):
            right_partition = parts[idx]
            # Sum the estimated sources to get the estimated mixtures
            ordered[b, :, :] = torch.stack(
                [est_targets[b, idx, :][None, :, :].sum(1) for idx in right_partition], dim=1
            )

        return ordered

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
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    separation_loss_weight : float, optional
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
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        max_num_speakers: int = None,  # deprecated in favor of `max_speakers_per_chunk``
        loss: Literal["bce", "mse"] = None,  # deprecated
        separation_loss_weight: float = 0.5,
        original_mixtures_for_separation: bool = False,
        forced_alignment_weight: float = 0.0,
        force_alignment: bool = False,
        log_alignment_accuracy: bool = False,
    ):
        super().__init__(
            protocol,
            duration=duration,
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

        if batch_size % 2 != 0:
            raise ValueError(
                "`batch_size` must be divisible by 2 for mixtures of mixtures training"
            )

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.weigh_by_cardinality = weigh_by_cardinality
        self.balance = balance
        self.weight = weight
        self.separation_loss = CustomMixITLossWrapper(multisrc_neg_sisdr, generalized=True)
        self.pit_sep_loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.separation_loss_weight = separation_loss_weight
        self.force_alignment = force_alignment
        self.original_mixtures_for_separation = original_mixtures_for_separation
        self.forced_alignment_weight = forced_alignment_weight
        self.log_alignment_accuracy = log_alignment_accuracy

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
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
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

        # gather all annotations of current file
        annotations = self.annotations[self.annotations["file_id"] == file_id]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output and input resolutions
        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start
        start_idx = np.floor(start / self.model.example_output[0].frames.step).astype(
            int
        )
        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start
        end_idx = np.ceil(end / self.model.example_output[0].frames.step).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_annotations[label_scope_key]))
        num_labels = len(labels)

        if num_labels > self.max_speakers_per_chunk:
            pass

        # initial frame-level targets
        y = np.zeros(
            (self.model.example_output[0].num_frames, num_labels), dtype=np.uint8
        )

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

        if self.original_mixtures_for_separation:
            start_idx_samples = np.floor(start * 16000).astype(int)
            end_idx_samples = np.floor(end * 16000).astype(int)
            sample_level_labels = np.zeros((num_samples, num_labels), dtype=np.uint8)
            for start, end, label in zip(
                start_idx_samples, end_idx_samples, chunk_annotations[label_scope_key]
            ):
                mapped_label = mapping[label]
                sample_level_labels[start:end, mapped_label] = 1

            # only frames with a single label should be used for mixit training
            sample["X_separation_mask"] = torch.from_numpy(
                sample_level_labels.sum(axis=1) == 1
            )

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
            annotations = self.annotations[
                np.where(self.annotations["file_id"] == file_id)[0]
            ]

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
                        annotated_region_indices, p=prob_annotated_regions_duration
                    )

                    # select one chunk at random in this annotated region
                    _, _, start, end = self.annotated_regions[annotated_region_index]
                    start_time = rng.uniform(start, end - duration)

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
                    previous_time = self.annotated_regions["start"][
                        annotated_region_indices[0]
                    ]
                    for segment in merged_repeated_segments:
                        if (
                            segment[0]
                            > self.annotated_regions["end"][
                                annotated_region_indices[current_region_index]
                            ]
                        ):
                            current_region_index += 1
                            previous_time = self.annotated_regions["start"][
                                annotated_region_indices[current_region_index]
                            ]

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

        if self.original_mixtures_for_separation:
            collated_X_separation_mask = self.collate_X_separation_mask(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y.unsqueeze(1),
        )

        if self.original_mixtures_for_separation:
            return {
                "X": augmented.samples,
                "y": augmented.targets.squeeze(1),
                "meta": collated_meta,
                "X_separation_mask": collated_X_separation_mask,
            }
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

        # forward pass
        bsz = waveform.shape[0]
        num_samples = waveform.shape[2]
        mix1 = waveform[0::2].squeeze(1)
        mix2 = waveform[1::2].squeeze(1)
        if self.original_mixtures_for_separation:
            # extract parts with only one speaker from original mixtures
            mix1_masks = batch["X_separation_mask"][0::2]
            mix2_masks = batch["X_separation_mask"][1::2]
            mix1_masked = mix1 * mix1_masks
            mix2_masked = mix2 * mix2_masks

        (
            mom,
            mom_target,
            num_active_speakers_mix1,
            num_active_speakers_mix2,
        ) = self.create_mixtures_of_mixtures(mix1, mix2, target[0::2], target[1::2])
        target = torch.cat((target[0::2], target[1::2], mom_target), dim=0)

        diarization, sources = self.model(torch.cat((mix1, mix2, mom), dim=0))
        mix1_sources = sources[: bsz // 2]
        mix2_sources = sources[bsz // 2 : bsz]
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

        if self.specifications[0].powerset:
            raise NotImplementedError("Forced alignment requires multilabel diarization")

        else:
            # last 2 sources should only contain noise so we force diarization outputs to 0
            permutated_diarization, permutations = permutate(target, diarization[:, :, :3])
            target = torch.cat((target, torch.zeros(batch_size, num_frames, 2, device=target.device)), dim=2)
            permutated_diarization = torch.cat((permutated_diarization, diarization[:, :, 3:]), dim=2)
            seg_loss = self.segmentation_loss(
                permutated_diarization, target, weight=weight
            )

        if self.force_alignment:
            speaker_idx_mix1 = [
                [permutations[i][j] for j in range(num_active_speakers_mix1[i])]
                for i in range(bsz // 2)
            ]
            speaker_idx_mix2 = [
                [
                    permutations[i][j]
                    for j in range(num_active_speakers_mix1[i], num_active_speakers_mix1[i] + num_active_speakers_mix2[i])
                ]
                for i in range(bsz // 2)
            ]
            
            est_mixes = []
            for i in range(bsz // 2):
                est_mix1 = mom_sources[i, :, speaker_idx_mix1[i]].sum(1) + mom_sources[i,:,3]
                est_mix2 = mom_sources[i, :, speaker_idx_mix2[i]].sum(1) + mom_sources[i,:,4]
                est_mix3 = mom_sources[i, :, speaker_idx_mix1[i]].sum(1) + mom_sources[i,:,4]
                est_mix4 = mom_sources[i, :, speaker_idx_mix2[i]].sum(1) + mom_sources[i,:,3]
                sep_loss_first_part = self.pit_sep_loss(
                    torch.stack((est_mix1, est_mix2)).unsqueeze(0), torch.stack((mix1[i], mix2[i])).unsqueeze(0)
                )
                sep_loss_second_part  = self.pit_sep_loss(
                    torch.stack((est_mix3, est_mix4)).unsqueeze(0), torch.stack((mix1[i], mix2[i])).unsqueeze(0)
                )
                if sep_loss_first_part < sep_loss_second_part:
                    est_mixes.append(torch.stack((est_mix1, est_mix2)))
                else:
                    est_mixes.append(torch.stack((est_mix3, est_mix4)))
            est_mixes = torch.stack(est_mixes)
            separation_loss = self.pit_sep_loss(
                est_mixes, torch.stack((mix1, mix2)).transpose(0, 1)
            ).mean()
            _, mixit_partitions = self.separation_loss(mom_sources[:,:,:3].transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1))
        else:
            separation_loss, _ = self.separation_loss(mom_sources[:,:,:3].transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1))

        if self.original_mixtures_for_separation:
            raise NotImplementedError
            # separation_loss += self.separation_loss(
            #     predicted_sources_mix1.transpose(1, 2), torch.stack((mix1_masked, torch.zeros_like(mix1))).transpose(0, 1), speaker_idx_mix1[0::3], speaker_idx_mix2[0::3]
            # ) * mix1_masks.sum() / num_samples / bsz * 3 + self.separation_loss(
            #     predicted_sources_mix2.transpose(1, 2), torch.stack((mix2_masked, torch.zeros_like(mix2))).transpose(0, 1), speaker_idx_mix1[1::3], speaker_idx_mix2[1::3]
            # ) * mix2_masks.sum() / num_samples / bsz * 3

        # forced_alignment_loss = (
        #     (1 - 2 * upscaled_permutated_target[: bsz // 2]) * mix1_sources**2
        #     + (1 - 2 * upscaled_permutated_target[bsz // 2 : bsz]) * mix2_sources**2
        #     + (1 - 2 * upscaled_permutated_target[bsz:]) * mom_sources**2
        # )
        # forced_alignment_loss = forced_alignment_loss.mean() / 3
        forced_alignment_loss = 0
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
            (1 - self.separation_loss_weight) * seg_loss
            + self.separation_loss_weight * separation_loss
            + forced_alignment_loss * self.forced_alignment_weight
        )

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
        if self.log_alignment_accuracy and self.force_alignment:
            for i in range(bsz // 2):
                inverse_mixit_partition = permutations_inverse[i][mixit_partitions[i][0]], permutations_inverse[i][mixit_partitions[i][1]]
                if set([int(j) for j in speaker_idx_mix1[i]]) <= set(inverse_mixit_partition[0].tolist()) and set([int(j) for j in speaker_idx_mix2[i]]) <= set(inverse_mixit_partition[1].tolist()):
                    self.num_correct += 1
                    if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(1, 0), (0, 1)]:
                        self.num_correct10 += 1
                    if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(2, 0), (0, 2)]:
                        self.num_correct20 += 1
                    if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(3, 0), (0, 3)]:
                        self.num_correct30 += 1
                    if num_active_speakers_mix1[i] == 1 and num_active_speakers_mix2[i] == 1:
                        self.num_correct11 += 1
                    if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(1, 2), (2, 1)]:
                        self.num_correct21 += 1
                if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(1, 0), (0, 1)]:
                    self.num_total10 += 1
                if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(2, 0), (0, 2)]:
                    self.num_total20 += 1
                if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(3, 0), (0, 3)]:
                    self.num_total30 += 1
                if num_active_speakers_mix1[i] == 1 and num_active_speakers_mix2[i] == 1:
                    self.num_total11 += 1
                if (num_active_speakers_mix1[i], num_active_speakers_mix2[i]) in [(1, 2), (2, 1)]:
                    self.num_total21 += 1
                self.num_total+=1
            if self.num_total30 > 0:
                self.model.log("accuracy/3_0", self.num_correct30/self.num_total30, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.num_total20 > 0:
                self.model.log("accuracy/2_0", self.num_correct20/self.num_total20, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.num_total10 > 0:
                self.model.log("accuracy/1_0", self.num_correct10/self.num_total10, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.num_total11 > 0:
                self.model.log("accuracy/1_1", self.num_correct11/self.num_total11, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.num_total21 > 0:
                self.model.log("accuracy/2_1", self.num_correct21/self.num_total21, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if self.num_total > 0:
                self.model.log("accuracy/total", self.num_correct/self.num_total, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/3_0", self.num_total30, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/2_0", self.num_total20, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/1_0", self.num_total10, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/1_1", self.num_total11, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/2_1", self.num_total21, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.model.log("counts/total", self.num_total, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
        num_samples = waveform.shape[2]
        # MoMs can't be created for batch size < 2
        if bsz < 2:
            return None
        # if bsz not even, then leave out last sample
        if bsz % 2 != 0:
            waveform = waveform[:-1]

        # if bsz not even, then leave out last sample
        mix1 = waveform[0::2].squeeze(1)
        mix2 = waveform[1::2].squeeze(1)
        if self.original_mixtures_for_separation:
            # extract parts with only one speaker from original mixtures
            mix1_masks = batch["X_separation_mask"][0::2]
            mix2_masks = batch["X_separation_mask"][1::2]
            mix1_masked = mix1 * mix1_masks
            mix2_masked = mix2 * mix2_masks

        (
            mom,
            mom_target,
            num_active_speakers_mix1,
            num_active_speakers_mix2,
        ) = self.create_mixtures_of_mixtures(mix1, mix2, target[0::2], target[1::2])

        # forward pass
        diarization, _ = self.model(waveform)
        _, mom_sources = self.model(mom)
        batch_size, num_frames, _ = diarization.shape

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        if self.specifications[0].powerset:
            raise NotImplementedError("Forced alignment requires multilabel diarization")

        else:
            # last 2 sources should only contain noise so we force diarization outputs to 0
            permutated_diarization, permutations = permutate(target, diarization[:, :, :3])
            target = torch.cat((target, torch.zeros(batch_size, num_frames, 2, device=target.device)), dim=2)
            permutated_diarization = torch.cat((permutated_diarization, diarization[:, :, 3:]), dim=2)
            seg_loss = self.segmentation_loss(
                permutated_diarization, target, weight=weight
            )

        if self.force_alignment:
            speaker_idx_mix1 = [
                [permutations[i][j] for j in range(num_active_speakers_mix1[i])]
                for i in range(bsz // 2)
            ]
            speaker_idx_mix2 = [
                [
                    permutations[i][j]
                    for j in range(num_active_speakers_mix1[i], num_active_speakers_mix1[i] + num_active_speakers_mix2[i])
                ]
                for i in range(bsz // 2)
            ]
            
            est_mixes = []
            for i in range(bsz // 2):
                est_mix1 = mom_sources[i, :, speaker_idx_mix1[i]].sum(1) + mom_sources[i,:,3]
                est_mix2 = mom_sources[i, :, speaker_idx_mix2[i]].sum(1) + mom_sources[i,:,4]
                est_mix3 = mom_sources[i, :, speaker_idx_mix1[i]].sum(1) + mom_sources[i,:,4]
                est_mix4 = mom_sources[i, :, speaker_idx_mix2[i]].sum(1) + mom_sources[i,:,3]
                sep_loss_first_part = self.pit_sep_loss(
                    torch.stack((est_mix1, est_mix2)).unsqueeze(0), torch.stack((mix1[i], mix2[i])).unsqueeze(0)
                )
                sep_loss_second_part  = self.pit_sep_loss(
                    torch.stack((est_mix3, est_mix4)).unsqueeze(0), torch.stack((mix1[i], mix2[i])).unsqueeze(0)
                )
                if sep_loss_first_part < sep_loss_second_part:
                    est_mixes.append(torch.stack((est_mix1, est_mix2)))
                else:
                    est_mixes.append(torch.stack((est_mix3, est_mix4)))
            est_mixes = torch.stack(est_mixes)
            separation_loss = self.pit_sep_loss(
                est_mixes, torch.stack((mix1, mix2)).transpose(0, 1)
            ).mean()
            _, mixit_partitions = self.separation_loss(mom_sources[:,:,:3].transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1))
        else:
            separation_loss, _ = self.separation_loss(mom_sources[:,:,:3].transpose(1, 2), torch.stack((mix1, mix2)).transpose(0, 1))

        if self.original_mixtures_for_separation:
            raise NotImplementedError

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

        if self.specifications[0].powerset:
            raise NotImplementedError("Forced alignment requires multilabel diarization")
        else:
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

        if self.specifications[0].powerset:
            raise NotImplementedError("Forced alignment requires multilabel diarization")
        else:
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
