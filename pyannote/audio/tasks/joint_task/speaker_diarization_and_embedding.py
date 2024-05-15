# MIT License
#
# Copyright (c) 2023- CNRS
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
import math
import random
import warnings
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_metric_learning.losses import ArcFaceLoss
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.utils.loss import nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker

Subtask = Literal["diarization", "embedding"]

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)
Subtasks = list(Subtask.__args__)


class JointSpeakerDiarizationAndEmbedding(SpeakerDiarization):
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
        weigh_by_cardinality: bool = False,
        batch_size: int = 32,
        dia_task_rate: float = 0.5,
        num_workers: int = None,
        pin_memory: bool = False,
        margin: float = 28.6,
        scale: float = 64.0,
        alpha: float = 0.5,
        augmentation: BaseWaveformTransform = None,
        cache: Optional[Union[str, None]] = None,
    ) -> None:
        """TODO Add docstring"""
        super().__init__(
            protocol,
            duration=duration,
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            weigh_by_cardinality=weigh_by_cardinality,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            cache=cache,
        )

        self.num_dia_samples = int(batch_size * dia_task_rate)
        self.margin = margin
        self.scale = scale
        self.alpha = alpha
        # keep track of the use of database available in the meta protocol
        # * embedding databases are those with global speaker label scope
        # * diarization databases are those with file or database speaker label scope
        self.embedding_files_id = []

    def setup(self, stage="fit"):
        """Setup method

        Parameters
        ----------
        stage : {'fit', 'validate', 'test'}, optional
            Setup stage. Defaults to 'fit'.
        """

        super().setup()

        global_scope_mask = (
            self.prepared_data["annotations-segments"]["global_label_idx"] > -1
        )
        self.embedding_files_id = np.unique(
            self.prepared_data["annotations-segments"]["file_id"][global_scope_mask]
        )
        embedding_classes = np.unique(
            self.prepared_data["annotations-segments"]["global_label_idx"][
                global_scope_mask
            ]
        )

        # if there is no file dedicated to the embedding task
        if self.alpha != 1.0 and len(embedding_classes) == 0:
            self.num_dia_samples = self.batch_size
            self.alpha = 1.0
            warnings.warn(
                "No class found for the speaker embedding task. Model will be trained on the speaker diarization task only."
            )

        if self.alpha != 0.0 and np.sum(global_scope_mask) == len(
            self.prepared_data["annotations-segments"]
        ):
            self.num_dia_samples = 0
            self.alpha = 0.0
            warnings.warn(
                "No segment found for the speaker diarization task. Model will be trained on the speaker embedding task only."
            )

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
            classes=embedding_classes,
        )
        self.specifications = (speaker_diarization, speaker_embedding)

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

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(
            file, chunk, duration=duration, mode="pad"
        )

        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field().step
        half = 0.5 * self.model.receptive_field().duration

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

    def draw_diarization_chunk(
        self,
        file_ids: np.ndarray,
        cum_prob_annotated_duration: np.ndarray,
        rng: random.Random,
        duration: float,
    ) -> tuple:
        """Sample one chunk for the diarization task

        Parameters
        ----------
        file_ids: np.ndarray
            array containing files id
        cum_prob_annotated_duration: np.ndarray
            array of the same size than file_ids array, containing probability
            to corresponding file to be drawn
        rng : random.Random
            Random number generator
        duration: float
            duration of the chunk to draw
        """
        # select one file at random (wiht probability proportional to its annotated duration)
        file_id = file_ids[cum_prob_annotated_duration.searchsorted(rng.random())]
        # find indices of annotated regions in this file
        annotated_region_indices = np.where(
            self.prepared_data["annotations-regions"]["file_id"] == file_id
        )[0]

        # turn annotated regions duration into a probability distribution
        cum_prob_annotaded_regions_duration = np.cumsum(
            self.prepared_data["annotations-regions"]["duration"][
                annotated_region_indices
            ]
            / np.sum(
                self.prepared_data["annotations-regions"]["duration"][
                    annotated_region_indices
                ]
            )
        )

        # seletect one annotated region at random (with probability proportional to its duration)
        annotated_region_index = annotated_region_indices[
            cum_prob_annotaded_regions_duration.searchsorted(rng.random())
        ]

        # select one chunk at random in this annotated region
        _, region_duration, start = self.prepared_data["annotations-regions"][
            annotated_region_index
        ]
        start_time = rng.uniform(start, start + region_duration - duration)

        return (file_id, start_time)

    def draw_embedding_chunk(self, class_id: int, duration: float) -> tuple:
        """Sample one chunk for the embedding task

        Parameters
        ----------
        class_id : int
            class ID in the task speficiations
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
        # get segments for current class
        class_segments_idx = (
            self.prepared_data["annotations-segments"]["global_label_idx"] == class_id
        )
        class_segments = self.prepared_data["annotations-segments"][class_segments_idx]

        # sample one segment from all the class segments:
        segments_duration = class_segments["end"] - class_segments["start"]
        segments_total_duration = np.sum(segments_duration)
        prob_segments = segments_duration / segments_total_duration
        segment = np.random.choice(class_segments, p=prob_segments)

        # sample chunk start time in order to intersect it with the sampled segment
        start_time = np.random.uniform(
            max(segment["start"] - duration, 0), segment["end"]
        )

        return (segment["file_id"], start_time)

    def train__iter__helper(self, rng: random.Random, **filters):
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

        # indices of training files that matches domain filters
        training = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "train"
        )
        for key, value in filters.items():
            training &= self.prepared_data["audio-metadata"][key] == self.prepared_data[
                "metadata"
            ][key].index(value)
        file_ids = np.where(training)[0]
        # get the subset of embedding database files from training files
        embedding_files_ids = file_ids[np.isin(file_ids, self.embedding_files_id)]

        if self.num_dia_samples > 0:
            annotated_duration = self.prepared_data["audio-annotated"][file_ids]
            # set duration of files for the embedding part to zero, in order to not
            # drawn them for diarization part
            annotated_duration[embedding_files_ids] = 0.0

            cum_prob_annotated_duration = np.cumsum(
                annotated_duration / np.sum(annotated_duration)
            )

        duration = self.duration
        batch_size = self.batch_size

        # use original order for the first run on the shuffled classes list:
        emb_task_classes = self.specifications[Subtasks.index("embedding")].classes[:]

        sample_idx = 0
        embedding_class_idx = 0
        while True:
            if sample_idx < self.num_dia_samples:
                file_id, start_time = self.draw_diarization_chunk(
                    file_ids, cum_prob_annotated_duration, rng, duration
                )
            else:
                # shuffle embedding classes list and go through this shuffled list
                # to make sure to see all the speakers during training
                if embedding_class_idx == len(emb_task_classes):
                    rng.shuffle(emb_task_classes)
                    embedding_class_idx = 0
                klass = emb_task_classes[embedding_class_idx]
                embedding_class_idx += 1
                file_id, start_time = self.draw_embedding_chunk(klass, duration)

            sample = self.prepare_chunk(file_id, start_time, duration)
            sample_idx = (sample_idx + 1) % batch_size

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
        rng = create_rng_for_worker(self.model)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)
        else:
            # create
            subchunks = dict()
            for product in itertools.product(
                [self.prepared_data["metadata-values"][key] for key in balance]
            ):
                filters = {key: value for key, value in zip(balance, product)}
                subchunks[product] = self.train__iter__helper(rng, **filters)

        while True:
            # select one subchunck generator at random (with uniform probability)
            # so thath it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(subchunks)]

            # generate random chunk
            yield next(chunks)

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

        collated_y_dia = []
        collate_y_emb = []

        for b in batch:
            # diarization reference
            y_dia = b["y"].data
            labels = b["y"].labels
            num_speakers = len(labels)
            # embedding reference
            y_emb = np.full((self.max_speakers_per_chunk,), -1, dtype=int)

            if num_speakers > self.max_speakers_per_chunk:
                # sort speakers in descending talkativeness order
                indices = np.argsort(-np.sum(y_dia, axis=0), axis=0)
                # keep only the most talkative speakers
                y_dia = y_dia[:, indices[: self.max_speakers_per_chunk]]
                # TODO: we should also sort the speaker labels in the same way

                # if current chunck is for the embedding subtask
                if b["meta"]["scope"] > 1:
                    labels = np.array(labels)
                    y_emb = labels[indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                # create inactive speakers by zero padding
                y_dia = np.pad(
                    y_dia,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )
                if b["meta"]["scope"] > 1:
                    y_emb[:num_speakers] = labels[:]

            else:
                if b["meta"]["scope"] > 1:
                    y_emb[:num_speakers] = labels[:]

            collated_y_dia.append(y_dia)
            collate_y_emb.append(y_emb)

        return (
            torch.from_numpy(np.stack(collated_y_dia)),
            torch.from_numpy(np.stack(collate_y_emb)).squeeze(1),
        )

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
        collated_y_dia, collate_y_emb = self.collate_y(batch)

        # collate metadata
        collated_meta = self.collate_meta(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y_dia.unsqueeze(1),
        )

        return {
            "X": augmented.samples,
            "y_dia": augmented.targets.squeeze(1),
            "y_emb": collate_y_emb,
            "meta": collated_meta,
        }

    def setup_loss_func(self):
        self.model.arc_face_loss = ArcFaceLoss(
            len(self.specifications[Subtasks.index("embedding")].classes),
            self.model.hparams["embedding_dim"],
            margin=self.margin,
            scale=self.scale,
        )

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

    def compute_diarization_loss(self, prediction, permutated_target):
        """Compute loss for the speaker diarization subtask

        Parameters
        ----------
        prediction : torch.Tensor
            speaker diarization output predicted by the model for the current batch.
            Shape of (batch_size, num_spk, num_frames)
        permutated_target: torch.Tensor
            permutated target for the current batch. Shape of (batch_size, num_spk, num_frames)

        Returns
        -------
        dia_loss : torch.Tensor
            Permutation-invariant diarization loss
        """

        # Compute segmentation loss
        dia_loss = self.segmentation_loss(prediction, permutated_target)
        self.model.log(
            "loss/train/dia",
            dia_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return dia_loss

    def compute_embedding_loss(self, emb_prediction, target_emb):
        """Compute loss for the speaker embeddings extraction subtask

        Parameters
        ----------
        emb_prediction : torch.Tensor
            speaker embeddings predicted by the model for the current batch.
            Shape of (batch_size * num_spk, embedding_dim)
        target_emb : torch.Tensor
            target embeddings for the current batch
            Shape of (batch_size * num_spk,)
        Returns
        -------
        emb_loss : torch.Tensor
            arcface loss for the current batch
        """

        # Get speaker representations from the embedding subtask
        embeddings = rearrange(emb_prediction, "b s e -> (b s) e")
        # Get corresponding target label
        targets = rearrange(target_emb, "b s -> (b s)")
        # compute loss only on global scope speaker embedding
        valid_emb = targets != -1

        # compute the loss
        emb_loss = self.model.arc_face_loss(
            embeddings[valid_emb, :], targets[valid_emb]
        )

        # skip batch if something went wrong for some reason
        if torch.isnan(emb_loss):
            return None

        self.model.log(
            "loss/train/arcface",
            emb_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return emb_loss

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
        alpha = self.alpha
        # batch waveforms (batch_size, num_channels, num_samples)
        waveform = batch["X"]
        # batch diarization references (batch_size, num_channels, num_speakers)
        target_dia = batch["y_dia"]
        # batch embedding references (batch, num_speakers)
        target_emb = batch["y_emb"]

        # drop samples that contain too many speakers
        num_speakers: torch.Tensor = torch.sum(torch.any(target_dia, dim=1), dim=1)
        keep: torch.Tensor = num_speakers <= self.max_speakers_per_chunk

        num_remaining_dia_samples = torch.sum(keep[: self.num_dia_samples])
        target_dia = target_dia[keep]
        target_emb = target_emb[keep]
        waveform = waveform[keep]

        # corner case
        if not keep.any():
            return None

        # forward pass
        dia_prediction, emb_prediction = self.model(waveform)
        # (batch_size, num_frames, num_cls), (batch_size, num_spk, emb_size)

        # get the best permutation
        dia_multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target_dia, permut_map = permutate(dia_multilabel, target_dia)
        permutated_target_emb = target_emb[
            torch.arange(target_emb.shape[0]).unsqueeze(1), permut_map
        ]

        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target_dia.float()
        )

        dia_prediction = dia_prediction[:num_remaining_dia_samples]
        permutated_target_powerset = permutated_target_powerset[
            :num_remaining_dia_samples
        ]

        dia_loss = torch.tensor(0)
        # if batch contains diarization subtask chunks, then compute diarization loss on these chunks:
        if self.alpha != 0.0 and torch.any(keep[: self.num_dia_samples]):
            dia_loss = self.compute_diarization_loss(
                dia_prediction, permutated_target_powerset
            )

        emb_loss = torch.tensor(0)
        # if batch contains embedding subtask chunks, then compute embedding loss on these chunks:
        if self.alpha != 1.0 and torch.any(keep[self.num_dia_samples :]):
            emb_prediction = emb_prediction[num_remaining_dia_samples:]
            permutated_target_emb = permutated_target_emb[num_remaining_dia_samples:]
            emb_loss = self.compute_embedding_loss(
                emb_prediction, permutated_target_emb
            )

        loss = alpha * dia_loss + (1 - alpha) * emb_loss
        return {"loss": loss}

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
        target_dia = batch["y_dia"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # TODO: should we handle validation samples with too many speakers
        # waveform = waveform[keep]
        # target = target[keep]

        # forward pass
        dia_prediction, _ = self.model(waveform)
        batch_size, num_frames, _ = dia_prediction.shape

        multilabel = self.model.powerset.to_multilabel(dia_prediction)
        permutated_target, _ = permutate(multilabel, target_dia)

        # FIXME: handle case where target have too many speakers?
        # since we don't need
        permutated_target_powerset = self.model.powerset.to_powerset(
            permutated_target.float()
        )
        seg_loss = self.segmentation_loss(
            dia_prediction,
            permutated_target_powerset,
        )

        self.model.log(
            "loss/val/dia",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target_dia, 1, 2),
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

        y = permutated_target.float().cpu().numpy()
        y_pred = multilabel.cpu().numpy()

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
            # "EqualErrorRate": EqualErrorRate(compute_on_cpu=True, distances=False),
            # "BinaryAUROC": BinaryAUROC(compute_on_cpu=True),
        }
