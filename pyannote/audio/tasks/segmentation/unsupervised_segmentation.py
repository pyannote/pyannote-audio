from typing import Any, Dict, Optional, OrderedDict, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.tasks import Segmentation


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        protocol: Protocol,
        teacher: Model,  # unsupervised param: model to use to generate truth
        use_pseudolabels: bool = True,  # generate pseudolabels in training mode
        augmentation_teacher: BaseWaveformTransform = None,
        pl_fw_passes: int = 1,  # how many forward passes to average to get the pseudolabels
        # supervised params
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
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
        """Unsupervised segmentation task.

        Parameters
        ----------
        protocol : Protocol
            pyannote.database protocol
        teacher : Model, optional
            Teacher model to use, will use the Task model if left unspecified. Defaults to None.
        use_pseudolabels : bool, optional
            Whether or not to use pseudolabels for training. Defaults to True.
        augmentation_teacher: BaseWaveformTransform, optional
            What augmentation to apply on the Teacher input. Defaults to None.
        pl_fw_passes : int, optional
            How many forward passes to average to get the pseudolabels. Defaults to 1.
        duration : float, optional
            Chunks duration. Defaults to 2s.
        max_num_speakers : int, optional
            Force maximum number of speakers per chunk (must be at least 2).
            Defaults to estimating it from the training set.
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
        loss : {"bce", "mse"}, optional
            Permutation-invariant segmentation loss. Defaults to "bce".
        vad_loss : {"bce", "mse"}, optional
            Add voice activity detection loss.
        metric : optional
            Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
            Defaults to AUROC (area under the ROC curve).
        """

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
            balance=balance,
            weight=weight,
            loss=loss,
            vad_loss=vad_loss,
            metric=metric,
        )

        if pl_fw_passes < 1:
            raise ValueError("pl_fw_passes must be strictly positive.")
        if pl_fw_passes > 1:
            raise ValueError(
                "pl_fw_passes for multiple forward passes isn't properly implemented yet "
            )
        if teacher is None:
            raise ValueError(
                "Using the model as its own teacher isn't supported yet. Please pass a teacher model."
            )

        self.teacher = teacher
        self.use_pseudolabels = use_pseudolabels
        self.augmentation_teacher = augmentation_teacher
        self.pl_fw_passes = pl_fw_passes

        self.teacher.eval()

    def get_teacher_outputs_passes(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the teacher output on the input given an augmentation.
        Handles averaging multiple forward passes (each with the augmentation reapplied).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        aug : BaseWaveformTransform
            Input augmentation
        fw_passes : int, optional
            Number of forward passes to apply to get the final output, by default 1

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The tuple :
            - y, the final output, of size (batch_size, num_frames, num_speakers)
            - y_passes, a tensor of all passes, of size (fw_passes, batch_size, num_frames, num_speakers)
        """
        out_fw_passes = []
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            # for each forward pass
            for i in range(fw_passes):
                teacher_input = x

                # Augment input if necessary
                if aug is not None:
                    augmented = aug(
                        samples=teacher_input,
                        sample_rate=self.model.hparams.sample_rate,
                    )
                    teacher_input = augmented.samples
                # Compute pseudolabels and detach to avoid "memory leaks"
                pl = self.teacher(waveforms=teacher_input).detach()
                out_fw_passes.append(pl)
            # compute mean of forward passes if needed, and round to make pseudolabels
            # TODO: make it work properly by permutating the forward passes so that they "agree"
            stacked_passes = torch.stack(out_fw_passes)
            if fw_passes == 1:
                out = out_fw_passes[0]
            else:
                out = torch.mean(stacked_passes, dim=0)
            out = torch.round(out).type(torch.int8)

        return out, stacked_passes

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ) -> torch.Tensor:
        out, _ = self.get_teacher_outputs_passes(x, aug, fw_passes)
        return out

    def collate_fn(self, batch, stage="train"):
        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)
        collated_batch = {"X": collated_X, "y": collated_y}

        if stage == "val":
            return collated_batch

        # Generate pseudolabels with teacher if necessary
        if self.use_pseudolabels:
            x = collated_X
            # compute pseudo labels
            pseudo_y = self.get_teacher_output(
                x=x, aug=self.augmentation_teacher, fw_passes=self.pl_fw_passes
            )
            collated_batch["y"] = pseudo_y

        # Augment x/pseudo y if an augmentation is specified
        if self.augmentation is not None:
            augmented = self.augmentation(
                samples=collated_batch["X"],
                sample_rate=self.model.hparams.sample_rate,
                targets=collated_batch["y"].unsqueeze(1),
            )
            collated_batch["X"] = augmented.samples
            collated_batch["y"] = augmented.targets.squeeze(1)

        return collated_batch


class TeacherEmaUpdate(Callback):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
        update_rate: float = 0.99,
    ):
        """Exponential moving average of weights.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When should the update happen, by default "epoch"
        update_interval : int, optional
            Update will happen every 'update_interval' epochs/batches, by default 1
        update_rate : float, optional
            How much to keep of the old weights each update. 0=instant copy, 1=never update weights. By default 0.99.
        """

        super().__init__()

        self.when = when
        if self.when != "epoch" and self.when != "batch":
            raise ValueError(
                "TeacherUpdate 'when' argument can only be 'epoch' or 'batch'"
            )

        if update_rate < 0.0 or update_rate > 1.0:
            raise ValueError(
                f"Illegal update rate value ({update_rate}), it should be in [0.0,1.0]"
            )

        self.update_interval = update_interval
        self.update_rate = update_rate
        self.teacher_weights = None

    @staticmethod
    def get_decayed_weights(
        teacher_w: OrderedDict[str, torch.Tensor],
        student_w: OrderedDict[str, torch.Tensor],
        tau: float,
    ):
        with torch.no_grad():
            return {
                k: teacher_w[k] * tau + student_w[k].to(teacher_w[k].device) * (1 - tau)
                for k in student_w.keys()
            }

    # --- Methods that get called when necessary

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.teacher_weights = initial_model_w

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.teacher_weights

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        if progress % self.update_interval != 0:
            return False

        self.teacher_weights = TeacherEmaUpdate.get_decayed_weights(
            teacher_w=self.teacher_weights,
            student_w=current_model.state_dict(),
            tau=self.update_rate,
        )
        return True

    # --- Callback hooks

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
            self.update_teacher_and_cache(batch_idx, trainer, pl_module)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.when == "epoch":
            self.update_teacher_and_cache(trainer.current_epoch, trainer, pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        initial_teacher_w = pl_module.task.teacher.state_dict()
        self._setup_initial(initial_teacher_w)

    # --- Generic logic called from hooks

    def update_teacher_and_cache(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        # will indicate if teacher cache needs to be updated
        cache_dirty = self._teacher_update_state(progress, model)

        # If a weight has changed, compute updated teacher weights, cache it, and assign it
        if cache_dirty:
            try:
                new_teacher_w = self._compute_teacher_weights()
                model.task.teacher.load_state_dict(new_teacher_w)
            except AttributeError as err:
                raise AttributeError(
                    f"TeacherUpdate callback can't be applied on this model : {err}"
                )
