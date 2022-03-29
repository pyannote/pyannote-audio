from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Text, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pyannote.core import Segment
from pyannote.database import Protocol
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task, ValDataset
from pyannote.audio.tasks import Segmentation
from pyannote.audio.torchmetrics.functional.audio.diarization_error_rate import (
    diarization_error_rate,
)
from pyannote.audio.utils.permutation import permutate


class PseudoLabelPostprocess:
    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class UnsupervisedSegmentation(Segmentation, Task):
    OVERLAP_DEFAULTS = {"probability": 0.0, "snr_min": 0.0, "snr_max": 10.0}

    def __init__(
        self,
        model: Model,  # unsupervised param: model to use to generate truth
        protocol: Protocol,
        fake_in_train=True,  # generate fake truth in training mode
        fake_in_val=True,  # generate fake truth in val mode
        augmentation_model: BaseWaveformTransform = None,
        pl_postprocess: Union[
            PseudoLabelPostprocess, List[PseudoLabelPostprocess]
        ] = None,
        pl_fw_passes: int = 1,  # how many forward passes to average to get the pseudolabels
        val_fw_passes: int = 1,  # how many forward passes to average to get the validation uncertainty
        val_augmentation: BaseWaveformTransform = None,
        # supervised params
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        overlap: dict = OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        loss: Literal["bce", "mse"] = "bce",
        vad_loss: Literal["bce", "mse"] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):
        super().__init__(
            # Mixin params
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            # Segmentation params
            max_num_speakers=max_num_speakers,
            overlap=overlap,
            balance=balance,
            weight=weight,
            loss=loss,
            vad_loss=vad_loss,
            metric=metric,
        )

        self.teacher = model
        self.fake_in_train = fake_in_train
        self.fake_in_val = fake_in_val
        self.augmentation_model = augmentation_model
        self.pl_fw_passes = pl_fw_passes
        self.val_fw_passes = val_fw_passes
        self.val_augmentation = val_augmentation

        if pl_fw_passes > 1 and augmentation_model is None:
            raise ValueError(
                "There is no reason to do multiple forward passes to generate pseudolabels if there is no augmentation applied to the input."
            )

        if val_augmentation is None and val_fw_passes > 1:
            raise ValueError(
                f"You need to specify a validation augmentation if you want to do {val_fw_passes} val forward passes."
            )

        if isinstance(pl_postprocess, PseudoLabelPostprocess):
            self.pl_postprocess = [pl_postprocess]
        else:
            self.pl_postprocess = pl_postprocess

        self.teacher.eval()

    def get_teacher_outputs(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out_fw_passes = []
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            for i in range(fw_passes):
                teacher_input = x

                if aug is not None:
                    teacher_input = aug(
                        teacher_input, sample_rate=self.model.hparams.sample_rate
                    )

                    # detach is necessary to avoid memory leaks
                    pl = self.teacher(waveforms=x).detach()
                out_fw_passes.append(pl)

            out = torch.mean(torch.stack(out_fw_passes), dim=0)
            out = torch.round(out).type(torch.int8)

        return out, torch.stack(out_fw_passes)

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out, _ = self.get_teacher_outputs(x, aug, fw_passes)
        return out

    def use_pseudolabels(self, stage: Literal["train", "val"]):
        return (stage == "train" and self.fake_in_train) or (
            stage == "val" and self.fake_in_val
        )

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with teacher if they are not provided
        if self.use_pseudolabels("train"):
            x = collated_batch["X"]

            pseudo_y, computed_ys = self.get_teacher_outputs(
                x=x, aug=self.augmentation_model, fw_passes=self.pl_fw_passes
            )

            y = None
            if "y" in collated_batch:
                y = collated_batch["y"]
            if self.pl_postprocess is not None:
                for pp in self.pl_postprocess:
                    pseudo_y, x = pp.process(pseudo_y, y, x, computed_ys)

            collated_batch["y"] = pseudo_y
            collated_batch["X"] = x

        if self.augmentation is not None:
            collated_batch["X"] = self.augmentation(
                collated_batch["X"], sample_rate=self.model.hparams.sample_rate
            )
        return collated_batch

    def collate_fn_val(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with teacher if they are not provided
        if self.use_pseudolabels("val"):
            teacher_input = collated_batch["X"]
            collated_batch["y"] = self.get_teacher_output(x=teacher_input)
            collated_batch = torch.round(collated_batch).type(torch.int8)

        return collated_batch

    def prepare_chunk(
        self,
        file: AudioFile,
        chunk: Segment,
        duration: float = None,
        stage: Literal["train", "val"] = "train",
    ) -> Tuple[np.ndarray, np.ndarray, List[Text]]:
        """Extract audio chunk and corresponding frame-wise labels

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment
            Audio chunk.
        duration : float, optional
            Fix chunk duration to avoid rounding errors. Defaults to self.duration
        stage : {"train", "val"}
            "train" for training step, "val" for validation step

        Returns
        -------
        sample : dict
            Dictionary with the following keys:
            X : np.ndarray
                Audio chunk as (num_samples, num_channels) array.
            y : np.ndarray
                Frame-wise labels as (num_frames, num_labels) array.
            ...
        """

        sample = super().prepare_chunk(
            file, chunk, duration=duration, stage=stage, use_annotations=True
        )
        return sample

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.has_validation:
            return DataLoader(
                ValDataset(self),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                collate_fn=self.collate_fn_val,
            )
        else:
            return None

    def validation_step(self, batch, batch_idx: int):
        super().validation_step(batch, batch_idx)

        x = batch["X"]

        torch.set_grad_enabled(True)
        self.model.train()
        self.model.zero_grad()
        prediction = self.model(x)
        teacher_prediction = self.teacher(x.to(self.teacher.device)).to(
            self.model.device
        )
        teacher_prediction = (teacher_prediction > 0.5).float()
        permutated_prediction, _ = permutate(teacher_prediction, prediction)
        loss = self.segmentation_loss(permutated_prediction, teacher_prediction)
        loss.backward()

        summed_grad_norm = 0
        for p in self.model.parameters():
            summed_grad_norm += torch.norm(p.grad)

        self.model.zero_grad()
        self.teacher.zero_grad()
        self.model.eval()
        torch.set_grad_enabled(False)
        self.model.log(
            f"{self.logging_prefix}GradientsSummedNorm",
            summed_grad_norm,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        outputs = []
        if self.val_augmentation is not None:
            for i in range(self.val_fw_passes):
                pred = self.model(
                    self.val_augmentation(x, sample_rate=self.model.hparams.sample_rate)
                )
                outputs.append(pred)
        else:
            outputs.append(prediction)
        outputs = torch.stack(outputs)

        # Compute ang log uncertainty
        std = torch.std(outputs, dim=0)
        self.model.log(
            f"{self.logging_prefix}Uncertainty",
            std.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Compute and log confidence
        outputs_mean = torch.mean(outputs, dim=0)
        CONFIDENCE_CENTER = 0.5
        confidence = 1 - torch.abs(CONFIDENCE_CENTER - outputs_mean) / torch.where(
            outputs_mean > CONFIDENCE_CENTER, 1 - CONFIDENCE_CENTER, CONFIDENCE_CENTER
        )
        self.model.log(
            f"{self.logging_prefix}Confidence",
            confidence.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        CONFIDENCE_CENTER = 0.25
        confidence = 1 - torch.abs(CONFIDENCE_CENTER - outputs_mean) / torch.where(
            outputs_mean > CONFIDENCE_CENTER, 1 - CONFIDENCE_CENTER, CONFIDENCE_CENTER
        )

        self.model.log(
            f"{self.logging_prefix}Confidence25",
            confidence.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


def _compute_ders(
    pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor]:
    batch_size = pseudo_y.shape[0]
    ders = torch.zeros(batch_size)

    tm_pseudo_y = pseudo_y.swapaxes(1, 2)
    tm_true_y = y.swapaxes(1, 2)
    for i in range(batch_size):
        ders[i] = diarization_error_rate(
            tm_pseudo_y[i][None, :, :], tm_true_y[i][None, :, :]
        )

    return ders


class DiscardConfidence(PseudoLabelPostprocess):
    def __init__(self, threshold=0.75) -> None:
        super().__init__()
        self.threshold = threshold

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ys is (fw_passes, batch_size, num_frames, num_speakers)-shaped
        fw_passes, batch_size, num_frames, num_speakers = ys.shape

        # ys_confidence is of shape (batch_size)
        ys_confidence = torch.mean(
            torch.abs(
                torch.mean(ys, dim=0).reshape(batch_size, num_frames * num_speakers)
                - 0.5
            )
            / 0.5,
            dim=1,
        )

        filter = ys_confidence > self.threshold
        return pseudo_y[filter], x[filter]


class DiscardPercentDer(PseudoLabelPostprocess):
    def __init__(self, ratio_to_discard: float = 0.1) -> None:
        self.ratio_to_discard = ratio_to_discard

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = pseudo_y.shape[0]
        ders = _compute_ders(pseudo_y, y, x)
        sorted_ders, sorted_indices = torch.sort(ders)

        to_discard_count = min(
            batch_size, max(1, round(batch_size * self.ratio_to_discard))
        )
        pseudo_y = pseudo_y[sorted_indices][:-to_discard_count, :, :]
        x = x[sorted_indices][:-to_discard_count, :, :]

        return pseudo_y, x


class DiscardThresholdDer(PseudoLabelPostprocess):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ders = _compute_ders(pseudo_y, y, x)

        filter = torch.where(ders < self.threshold)
        pseudo_y = pseudo_y[filter]
        x = x[filter]

        return pseudo_y, x


class TeacherUpdate(Callback):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
        weight_update_rate: float = 0.0,
        average_of: int = 1,
    ):
        self.when = when
        self.update_interval = update_interval
        self.weight_update_rate = weight_update_rate
        self.average_of = average_of

        self.last_weights: List[OrderedDict[str, torch.Tensor]] = []
        self.teacher_weights_cache: OrderedDict[str, torch.Tensor] = None

    def enqueue_teacher(self, teacher: OrderedDict[str, torch.Tensor]):
        if len(self.last_weights) >= self.average_of:
            self.last_weights.pop(0)
            self.last_weights.append(teacher)

    def get_updated_weights(
        self,
        teacher_w: OrderedDict[str, torch.Tensor],
        student_w: OrderedDict[str, torch.Tensor],
    ):
        with torch.no_grad():
            return {
                k: teacher_w[k].to("cpu") * self.weight_update_rate
                + student_w[k].to("cpu") * (1 - self.weight_update_rate)
                for k in student_w
            }

    def compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        if len(self.last_weights) == 1:
            return self.last_weights[0]
        else:
            with torch.no_grad():
                new_w = {
                    k: torch.mean(torch.stack([w[k] for w in self.last_weights]), dim=0)
                    for k in self.last_weights[0]
                }
                return new_w

    def try_update_teacher(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        if (
            self.update_interval > 0
            and self.weight_update_rate < 1.0
            and progress % self.update_interval == 0
        ):
            try:
                # Get new teacher "candidate" (from decayed weights) and enqueue it in the teacher history
                teacher_candidate_w = self.get_updated_weights(
                    self.teacher_weights_cache, model.state_dict()
                )
                self.enqueue_teacher(teacher_candidate_w)

                # Compute the real new teacher weights, cache it, and assign it
                new_teacher_w = self.compute_teacher_weights()
                self.teacher_weights_cache = new_teacher_w
                model.task.teacher.load_state_dict(new_teacher_w)

            except AttributeError as err:
                print(f"TeacherUpdate callback can't be applied on this model : {err}")

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if len(self.last_weights) == 0:
            self.last_weights.append(pl_module.task.teacher.state_dict())
            self.teacher_weights_cache = pl_module.task.teacher.state_dict()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.when == "batch":
            self.try_update_teacher(batch_idx, trainer, pl_module)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.when == "epoch":
            self.try_update_teacher(trainer.current_epoch, trainer, pl_module)
