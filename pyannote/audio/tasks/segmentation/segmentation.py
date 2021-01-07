# MIT License
#
# Copyright (c) 2020 CNRS
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


import random
from typing import Callable, Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import auroc
from torch.nn import Parameter
from torch.optim import Optimizer
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.io import Audio
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class Segmentation(SegmentationTaskMixin, Task):
    """Segmentation

    Note that data augmentation is used to increase the proportion of "overlap".
    This is achieved by generating chunks made out of the (weighted) sum of two
    random chunks.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    num_speakers : int, optional
        Maximum number of speakers per chunk. Defaults to 4. Note that one should account
        for artificial chunks (see below) when setting this number.
    augmentation_probability : float, optional
        Probability of artificial overlapping chunks. A probability of 0.6 means that,
        on average, 40% of training chunks are "real" chunks, while 60% are artifical
        chunks made out of the (weighted) sum of two chunks. Defaults to 0.5.
    snr_min, snr_max : float, optional
        Minimum and maximum signal-to-noise ratio between summed chunks, in dB.
        Defaults to 0.0 and 10.
    domain : str, optional
        Indicate that data augmentation will only overlap chunks from the same domain
        (i.e. share the same file[domain] value). Default behavior is to not contrain
        data augmentation with regards to domain.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    vad_loss : {"bce", "mse"}, optional
        Add voice activity detection loss.
    """

    ACRONYM = "seg"

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        num_speakers: int = 4,
        augmentation_probability: float = 0.5,
        snr_min: float = 0.0,
        snr_max: float = 10.0,
        domain: str = None,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
        augmentation: BaseWaveformTransform = None,
        loss: Literal["bce", "mse"] = "bce",
        vad_loss: Literal["bce", "mse"] = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            optimizer=optimizer,
            learning_rate=learning_rate,
            augmentation=augmentation,
        )

        self.num_speakers = num_speakers

        self.augmentation_probability = augmentation_probability
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.domain = domain

        if loss not in ["bce", "mse"]:
            raise ValueError("'loss' must be one of {'bce', 'mse'}.")
        self.loss = loss
        self.vad_loss = vad_loss

        self.specifications = TaskSpecification(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            scale=Scale.FRAME,
            duration=self.duration,
            classes=[f"speaker#{i+1}" for i in range(self.num_speakers)],
            permutation_invariant=True,
        )

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            # build the list of domains
            if self.domain is not None:
                for f in self.train:
                    f["domain"] = f[self.domain]
                self.domains = list(set(f["domain"] for f in self.train))

    def setup_loss_func(self, model: Model):

        example_input_array = self.example_input_array
        _, _, num_samples = example_input_array.shape
        self.num_samples = num_samples

        batch_size, num_frames, num_speakers = model(example_input_array).shape
        self.num_frames = num_frames
        hamming_window = torch.hamming_window(num_frames, periodic=False).reshape(-1, 1)
        model.register_buffer("hamming_window", hamming_window)

        val_sample_weight = hamming_window.expand(
            batch_size, num_frames, num_speakers
        ).flatten()

        model.register_buffer("val_sample_weight", val_sample_weight)

    def prepare_y(self, one_hot_y: np.ndarray):
        """Get segmentation targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is two or more active speakers at tth frame, 0 otherwise.
        """

        num_frames, num_speakers = one_hot_y.shape

        # in case there are too many speakers, remove less talkative ones
        if num_speakers > self.num_speakers:

            # TODO: warn

            most_talkative = np.argpartition(
                np.sum(one_hot_y, axis=0), num_speakers - self.num_speakers
            )[-self.num_speakers :]

            one_hot_y = np.take(one_hot_y, most_talkative, axis=1)

        # in case there are not enough speakers, add empty ones
        elif num_speakers < self.num_speakers:
            one_hot_y = np.pad(
                one_hot_y, ((0, 0), (0, self.num_speakers - num_speakers))
            )

        return one_hot_y

    def train__iter__helper(self, rng: random.Random, domain: str = None):

        train = self.train

        if domain is not None:
            train = [f for f in train if f["domain"] == domain]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                train,
                weights=[f["duration"] for f in train],
                k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = rng.choices(
                file["annotated"],
                weights=[s.duration for s in file["annotated"]],
                k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = rng.uniform(segment.start, segment.end - self.duration)
            chunk = Segment(start_time, start_time + self.duration)

            yield self.prepare_chunk(file, chunk, duration=self.duration)

    def train__iter__(self):
        """Iterate over training samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: (frame, )
            Frame-level targets. Note that frame < time.
            `frame` is infered automagically from the
            example model output.
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.current_epoch)

        if self.domain is None:
            chunks = self.train__iter__helper(rng)
        else:
            chunks_by_domain = {
                domain: self.train__iter__helper(rng, domain=domain)
                for domain in self.domains
            }

        while True:

            if self.domain is not None:
                #  draw domain at random
                domain = rng.choice(self.domains)
                chunks = chunks_by_domain[domain]

            # generate random chunk
            X, one_hot_y, labels = next(chunks)

            if rng.random() > self.augmentation_probability:
                # yield it as it is
                yield {"X": X, "y": self.prepare_y(one_hot_y)}
                continue

            #  generate second random chunk
            other_X, other_one_hot_y, other_labels = next(chunks)

            #  sum both chunks with random SNR
            random_snr = (self.snr_max - self.snr_min) * rng.random() + self.snr_min
            alpha = np.exp(-np.log(10) * random_snr / 20)
            X = Audio.power_normalize(X) + alpha * Audio.power_normalize(other_X)

            # combine speaker-to-index mapping
            y_mapping = {label: i for i, label in enumerate(labels)}
            num_labels = len(y_mapping)
            for label in other_labels:
                if label not in y_mapping:
                    y_mapping[label] = num_labels
                    num_labels += 1

            #  combine one-hot-encoded speaker activities
            combined_y = np.zeros_like(one_hot_y, shape=(len(one_hot_y), num_labels))
            for i, label in enumerate(labels):
                combined_y[:, y_mapping[label]] += one_hot_y[:, i]
            for i, label in enumerate(other_labels):
                combined_y[:, y_mapping[label]] += other_one_hot_y[:, i]

            # handle corner case when the same label is active at the same time in both chunks
            combined_y = np.minimum(combined_y, 1, out=combined_y)

            yield {"X": X, "y": self.prepare_y(combined_y)}

    # def segmentation_loss(self, model, y, y_pred):
    def segmentation_loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        y : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        y_pred : torch.Tensor
            Speaker activations

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        permutated_y_pred, _ = permutate(y, y_pred)

        if self.loss == "bce":
            seg_losses = F.binary_cross_entropy(
                permutated_y_pred, y.float(), reduction="none"
            )

        elif self.loss == "mse":
            seg_losses = F.mse_loss(permutated_y_pred, y.float(), reduction="none")

        # seg_loss = torch.mean(seg_losses * model.hamming_window)
        seg_loss = torch.mean(seg_losses)

        return seg_loss

    def voice_activity_detection_loss(
        self, y: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:

        vad_y_pred, _ = torch.max(y_pred, dim=2, keepdim=False)
        vad_y, _ = torch.max(y.float(), dim=2, keepdim=False)

        if self.vad_loss == "bce":
            vad_losses = F.binary_cross_entropy(vad_y_pred, vad_y, reduction="none")

        elif self.vad_loss == "mse":
            vad_losses = F.mse_loss(vad_y_pred, vad_y, reduction="none")

        vad_loss = torch.mean(vad_losses)

        return vad_loss

    def training_step(self, model: Model, batch, batch_idx: int):
        """Compute permutation-invariant binary cross-entropy

        Parameters
        ----------
        model : Model
            Model currently being trained.
        batch : (usually) dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.

        Returns
        -------
        loss : {str: torch.tensor}
            {"loss": loss} with additional "loss_{task_name}" keys for multi-task models.
        """

        X, y = batch["X"], batch["y"]

        y_pred = model(X)
        # loss = self.segmentation_loss(model, y, y_pred)
        seg_loss = self.segmentation_loss(y, y_pred)

        model.log(
            f"{self.ACRONYM}@train_seg_loss",
            seg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            vad_loss = self.voice_activity_detection_loss(y, y_pred)

            model.log(
                f"{self.ACRONYM}@train_vad_loss",
                vad_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, model: Model, batch, batch_idx: int):
        """

        Parameters
        ----------
        model : Model
            Model currently being validated.
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        X, y = batch["X"], batch["y"]
        y_pred, _ = permutate(y, model(X))

        try:
            auc = auroc(
                y_pred.flatten(),
                y.flatten(),
                # give less importance to start and end of chunks
                # using the same (Hamming) window as inference.
                sample_weight=model.val_sample_weight,
                pos_label=1.0,
            )
        except ValueError:
            # in case of all positive or all negative samples, auroc will raise a ValueError.
            return

        model.log(
            f"{self.ACRONYM}@val_auroc",
            auc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
