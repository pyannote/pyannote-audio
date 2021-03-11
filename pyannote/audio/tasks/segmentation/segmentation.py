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
from collections import Counter
from typing import Text

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing_extensions import Literal

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.utils.loss import binary_cross_entropy, mse_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.core import SlidingWindow
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
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
    overlap: dict, optional
        Controls how artificial chunks with overlapping speech are generated:
        - "probability" key is the probability of artificial overlapping chunks. Setting
          "probability" to 0.6 means that, on average, 40% of training chunks are "real"
          chunks, while 60% are artifical chunks made out of the (weighted) sum of two
          chunks. Defaults to 0.5.
        - "snr_min" and "snr_max" keys control the minimum and maximum signal-to-noise
          ratio between summed chunks, in dB. Default to 0.0 and 10.
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
    vad_loss : {"bce", "mse"}, optional
        Add voice activity detection loss.
    """

    ACRONYM = "seg"

    OVERLAP_DEFAULTS = {"probability": 0.5, "snr_min": 0.0, "snr_max": 10.0}

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        overlap: dict = OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
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
            augmentation=augmentation,
        )

        self.overlap = overlap
        self.balance = balance
        self.weight = weight

        if loss not in ["bce", "mse"]:
            raise ValueError("'loss' must be one of {'bce', 'mse'}.")
        self.loss = loss
        self.vad_loss = vad_loss

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

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
                resolution=Resolution.FRAME,
                duration=self.duration,
                classes=[f"speaker#{i+1}" for i in range(self.num_speakers)],
                permutation_invariant=True,
            )

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

        if num_speakers > self.num_speakers:
            raise ValueError()

        if num_speakers < self.num_speakers:
            one_hot_y = np.pad(
                one_hot_y, ((0, 0), (0, self.num_speakers - num_speakers))
            )

        return one_hot_y

    def val__getitem__(self, idx):

        f, chunk = self._validation[idx]
        sample = self.prepare_chunk(f, chunk, duration=self.duration, stage="val")
        y, labels = sample["y"], sample.pop("labels")

        # since number of speakers is estimated from the training set,
        # we might encounter validation chunks that have more speakers.
        # in that case, we arbitrarily remove last speakers
        if y.shape[1] > self.num_speakers:
            y = y[:, : self.num_speakers]
            labels = labels[: self.num_speakers]

        sample["y"] = self.prepare_y(y)
        return sample

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

        if self.loss == "bce":
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
            )

        elif self.loss == "mse":
            seg_loss = mse_loss(permutated_prediction, target.float(), weight=weight)

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

        # forward pass
        prediction = self.model(batch["X"])
        # (batch_size, num_frames, num_classes)

        # target
        target = batch["y"]

        permutated_prediction, _ = permutate(target, prediction)

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(weight_key, None)
        # (batch_size, num_frames, 1)

        seg_loss = self.segmentation_loss(permutated_prediction, target, weight=weight)

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
            vad_loss = self.voice_activity_detection_loss(
                permutated_prediction, target, weight=weight
            )

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
