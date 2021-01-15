# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
import random
from collections import Counter
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Optimizer
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing_extensions import Literal

from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Scale, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment, SlidingWindow
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
        Defaults to multiprocessing.cpu_count() // 2.
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
        augmentation_probability: float = 0.5,
        snr_min: float = 0.0,
        snr_max: float = 10.0,
        domain: str = None,
        batch_size: int = 32,
        num_workers: int = None,
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

        self.augmentation_probability = augmentation_probability
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.domain = domain

        if loss not in ["bce", "mse"]:
            raise ValueError("'loss' must be one of {'bce', 'mse'}.")
        self.loss = loss
        self.vad_loss = vad_loss

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            # TODO: handle this domain thing in SegmentationTaskMixin

            # build the list of domains
            if self.domain is not None:
                for f in self._train:
                    f["domain"] = f[self.domain]
                self.domains = list(set(f["domain"] for f in self._train))

            # slide a window (with 1s step) over the whole training set
            # and keep track of the number of speakers in each location
            num_speakers = []
            for file in self._train:
                start = file["annotated"][0].start
                end = file["annotated"][-1].end
                window = SlidingWindow(
                    start=start,
                    end=end,
                    duration=self.duration,
                    step=1.0,
                )
                for chunk in window:
                    num_speakers.append(len(file["annotation"].crop(chunk).labels()))

            # because there might a few outliers, estimate the upper bound for the
            # number of speakers as the 99th percentile

            num_speakers, counts = zip(*list(Counter(num_speakers).items()))
            num_speakers, counts = np.array(num_speakers), np.array(counts)

            sorting_indices = np.argsort(num_speakers)
            num_speakers = num_speakers[sorting_indices]
            counts = counts[sorting_indices]

            self.num_speakers = num_speakers[
                np.where(np.cumsum(counts) / np.sum(counts) > 0.99)[0][0]
            ]

            # TODO: add a few more speakers to make sure we don't skip
            # too many artificial chunks (which might result in less
            # overlap that we think we have)

            # now that we know about the number of speakers upper bound
            # we can set task specifications
            self.specifications = Specifications(
                problem=Problem.MULTI_LABEL_CLASSIFICATION,
                scale=Scale.FRAME,
                duration=self.duration,
                classes=[f"speaker#{i+1}" for i in range(self.num_speakers)],
                permutation_invariant=True,
            )

    def setup_loss_func(self):

        example_input_array = self.model.example_input_array
        _, _, num_samples = example_input_array.shape
        self.num_samples = num_samples

        batch_size, num_frames, num_speakers = self.model(example_input_array).shape
        self.num_frames = num_frames
        hamming_window = torch.hamming_window(num_frames, periodic=False).reshape(-1, 1)
        self.model.register_buffer("hamming_window", hamming_window)

        val_sample_weight = hamming_window.expand(
            batch_size, num_frames, num_speakers
        ).flatten()

        self.model.register_buffer("val_sample_weight", val_sample_weight)

    def prepare_y(self, one_hot_y: np.ndarray):
        """Zero-pad segmentation targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        padded_one_hot_y : (num_frames, self.num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.
        """

        num_frames, num_speakers = one_hot_y.shape

        if num_speakers < self.num_speakers:
            one_hot_y = np.pad(
                one_hot_y, ((0, 0), (0, self.num_speakers - num_speakers))
            )

        return one_hot_y

    def train__iter__helper(self, rng: random.Random, domain: str = None):

        train = self._train

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
        rng = create_rng_for_worker(self.model.current_epoch)

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
                if one_hot_y.shape[1] > self.num_speakers:
                    # skip chunks that happen to have too many speakers
                    pass

                else:
                    # pad and yield good ones
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

            if combined_y.shape[1] > self.num_speakers:
                # skip artificial chunks that happen to have too many speakers
                pass

            else:
                # pad and yield good ones
                yield {"X": X, "y": self.prepare_y(combined_y)}

    def val__getitem__(self, idx):
        f, chunk = self._validation[idx]
        X, one_hot_y, _ = self.prepare_chunk(f, chunk, duration=self.duration)

        # since number of speakers is estimated from the training set,
        # we might encounter validation chunks that have more speakers.
        # in that case, we arbirarily remove last speakers
        if one_hot_y.shape[1] > self.num_speakers:
            one_hot_y = one_hot_y[:, : self.num_speakers]

        y = self.prepare_y(one_hot_y)

        return {"X": X, "y": y}

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

    def training_step(self, batch, batch_idx: int):
        """Compute permutation-invariant binary cross-entropy

        Parameters
        ----------
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

        y_pred = self.model(X)
        # loss = self.segmentation_loss(model, y, y_pred)
        seg_loss = self.segmentation_loss(y, y_pred)

        self.model.log(
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

            self.model.log(
                f"{self.ACRONYM}@train_vad_loss",
                vad_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        self.model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        """Compute validation F-score

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        # move metric to model device
        self.val_fbeta.to(self.model.device)

        X, y = batch["X"], batch["y"]
        # X = (batch_size, num_channels, num_samples)
        # y = (batch_size, num_frames, num_classes)

        y_pred = self.model(X)

        permutated_y_pred, _ = permutate(y, y_pred)

        # y_pred = (batch_size, num_frames, num_classes)

        val_fbeta = self.val_fbeta(
            permutated_y_pred[:, ::10].squeeze(), y[:, ::10].squeeze()
        )
        self.model.log(
            f"{self.ACRONYM}@val_fbeta",
            val_fbeta,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_idx > 0:
            return

        # visualize first 9 validation samples of first batch in Tensorboard
        X = X.cpu().numpy()
        y = y.float().cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        permutated_y_pred = permutated_y_pred.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=4 * nrows,
            ncols=ncols,
            figsize=(15, 10),
        )

        # reshape target so that there is one line per class when plottingit
        y[y == 0] = np.NaN
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):

            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot waveform
            ax_wav = axes[row_idx * 4 + 0, col_idx]
            sample_X = np.mean(X[sample_idx], axis=0)
            ax_wav.plot(sample_X)
            ax_wav.set_xlim(0, len(sample_X))
            ax_wav.get_xaxis().set_visible(False)
            ax_wav.get_yaxis().set_visible(False)

            # plot target
            ax_ref = axes[row_idx * 4 + 1, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot prediction
            ax_hyp = axes[row_idx * 4 + 2, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

            # plot permutated prediction
            ax_map = axes[row_idx * 4 + 3, col_idx]
            sample_y_pred_map = permutated_y_pred[sample_idx]
            ax_map.plot(sample_y_pred_map)
            ax_map.set_ylim(-0.1, 1.1)
            ax_map.set_xlim(0, len(sample_y))

        plt.tight_layout()

        self.model.logger.experiment.add_figure(
            f"{self.ACRONYM}@val_samples", fig, self.model.current_epoch
        )
