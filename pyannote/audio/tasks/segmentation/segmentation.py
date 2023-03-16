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
from collections import Counter
from typing import Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from pyannote.core import SlidingWindow
from pyannote.database import Protocol
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

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


class Segmentation(SegmentationTaskMixin, Task):
    """Speaker segmentation

    Parameters
    ----------
    protocol : Protocol
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
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
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
        protocol: Protocol,
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

        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self.weigh_by_cardinality = weigh_by_cardinality
        self.balance = balance
        self.weight = weight
        self.vad_loss = vad_loss

    def setup(self, stage: Optional[str] = None):

        super().setup(stage=stage)

        if self.max_speakers_per_chunk is None:

            # TODO: optimize this

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

            self.max_speakers_per_chunk = max(
                2,
                num_speakers[np.where(np.cumsum(counts) / np.sum(counts) > 0.99)[0][0]],
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
        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )

    def adapt_y(self, collated_y: torch.Tensor) -> torch.Tensor:
        """Get speaker diarization targets

        Parameters
        ----------
        collated_y : (batch_size, num_frames, num_speakers) tensor
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[b, f, s] = 1 if sth speaker is active at fth frame
                * one_hot_y[b, f, s] = 0 otherwise.

        Returns
        -------
        y : (batch_size, num_frames, max_speakers_per_chunk) tensor
            Same as collated_y, except we only keep ``max_speakers_per_chunk`` most
            talkative speakers (per sample).
        """

        batch_size, num_frames, num_speakers = collated_y.shape

        # maximum number of active speakers in a chunk
        max_speakers_per_chunk = max(
            1, torch.max(torch.sum(torch.sum(collated_y, dim=1) > 0.0, dim=1))
        )

        # sort speakers in descending talkativeness order
        indices = torch.argsort(torch.sum(collated_y, dim=1), dim=1, descending=True)

        # keep max_speakers_per_chunk most talkative speakers, for each chunk
        y = torch.zeros(
            (batch_size, num_frames, max_speakers_per_chunk), dtype=collated_y.dtype
        )
        for b, index in enumerate(indices):
            for k, i in zip(range(max_speakers_per_chunk), index):
                y[b, :, k] = collated_y[b, :, i.item()]

        return y

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

        if self.specifications.powerset:

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
            return {"loss": 0.0}

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

        if self.specifications.powerset:

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

        else:
            permutated_prediction, _ = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )

        self.model.log(
            f"{self.logging_prefix}TrainSegLoss",
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
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                f"{self.logging_prefix}TrainVADLoss",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        self.model.log(
            f"{self.logging_prefix}TrainLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss}

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns diarization error rate and its components"""

        if self.specifications.powerset:
            return [
                DiarizationErrorRate(0.5),
                SpeakerConfusionRate(0.5),
                MissedDetectionRate(0.5),
                FalseAlarmRate(0.5),
            ]

        return [
            OptimalDiarizationErrorRate(),
            OptimalDiarizationErrorRateThreshold(),
            OptimalSpeakerConfusionRate(),
            OptimalMissedDetectionRate(),
            OptimalFalseAlarmRate(),
        ]

    def train__iter__(self):
        for chunk in super().train__iter__():
            # TODO: document why this filtering is needed
            if self.specifications.powerset:
                if len(chunk["y"].labels) <= self.max_speakers_per_chunk:
                    yield chunk
            else:
                yield chunk

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

        # forward pass
        prediction = self.model(waveform)
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

        if self.specifications.powerset:

            powerset = torch.nn.functional.one_hot(
                torch.argmax(prediction, dim=-1),
                self.model.powerset.num_powerset_classes,
            ).float()
            multilabel = self.model.powerset.to_multilabel(powerset)
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

        self.model.log(
            f"{self.logging_prefix}ValSegLoss",
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
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                f"{self.logging_prefix}ValVADLoss",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        self.model.log(
            f"{self.logging_prefix}ValLoss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.specifications.powerset:
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

        if self.specifications.powerset:
            y = permutated_target_powerset.float().cpu().numpy()
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
                logger.experiment.add_figure(
                    f"{self.logging_prefix}ValSamples", fig, self.model.current_epoch
                )
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"{self.logging_prefix}ValSamples_epoch{self.model.current_epoch}.png",
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
