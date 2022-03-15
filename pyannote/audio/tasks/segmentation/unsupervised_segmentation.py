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


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        model: Model,  # unsupervised param: model to use to generate truth
        protocol: Protocol,
        fake_in_train=True,  # generate fake truth in training mode
        fake_in_val=True,  # generate fake truth in val mode
        augmentation_model: BaseWaveformTransform = None,
        # supervised params
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        overlap: dict = Segmentation.OVERLAP_DEFAULTS,
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

        self.teacher.eval()

    def get_model_output(self, model: Model, waveforms: torch.Tensor):
        result = None
        # try inference mode ?
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            result = model(
                waveforms=waveforms
            ).detach()  # detach is necessary to avoid memory leaks
            result = torch.round(result).type(torch.int8)
        return result

    def use_pseudolabels(self, stage: Literal["train", "val"]):
        return (stage == "train" and self.fake_in_train) or (
            stage == "val" and self.fake_in_val
        )

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with teacher if they are not provided
        if self.use_pseudolabels("train"):
            teacher_input = collated_batch["X"]
            if self.augmentation_model is not None:
                teacher_input = self.augmentation_model(
                    collated_batch["X"], sample_rate=self.model.hparams.sample_rate
                )
            collated_batch["y"] = self.get_model_output(self.teacher, teacher_input)

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
            collated_batch["y"] = self.get_model_output(self.teacher, teacher_input)

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
