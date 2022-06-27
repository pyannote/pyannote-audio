import warnings
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task, ValDataset
from pyannote.audio.tasks import Segmentation


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        protocol: Protocol,
        teacher: Model = None,  # unsupervised param: model to use to generate truth
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
            # compute mean of forward passes
            out = torch.mean(torch.stack(out_fw_passes), dim=0)
            out = torch.round(out).type(torch.int8)

        return out, torch.stack(out_fw_passes)

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out, _ = self.get_teacher_outputs_passes(x, aug, fw_passes)
        return out

    def collate_fn(self, batch, stage="train"):
        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)
        collated_batch = {"X": collated_X, "y": collated_y}

        if stage != "train":
            raise RuntimeError(f"Unexpected stage in collate_fn (stage={stage})")

        # Generate pseudolabels with teacher if necessary
        if self.use_pseudolabels:
            x = collated_X
            # compute pseudo labels
            pseudo_y, computed_y_passes = self.get_teacher_outputs_passes(
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

    def collate_fn_val(self, batch):
        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)

        collated_batch = {"X": collated_X, "y": collated_y}

        # Generate annotations y with teacher if they are not provided
        # TODO: reimplement or discard fake validation
        if self.use_pseudolabels:
            pass

        return collated_batch

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
    def __init__(self, when: Literal["epoch", "batch"]) -> None:
        """Abstract class for teacher update callback.
        _teacher_update_state, _compute_teacher_weights and _setup_initial needs to be overriden.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When the update is applied.
        """

        super().__init__()
        self.initialized = False

        self.when = when
        if self.when != "epoch" and self.when != "batch":
            raise ValueError(
                "TeacherUpdate 'when' argument can only be 'epoch' or 'batch'"
            )

        # The "real" teacher weights to be used by the task.
        # TODO: determine if useless ?
        self.cache: OrderedDict[str, torch.Tensor] = None

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        """Called every time the teacher might need to be updated.

        Parameters
        ----------
        progress : int
            Describes how far we are in training (epoch number or batch number)
        current_model : pl.LightningModule
            Current model being trained.

        Returns
        -------
        bool
            Whether or not the internal state has been updated.
            If true, the 'teacher' attribute of the model will be updated with
            newly computed weights from _compute_teacher_weights.
        """
        raise NotImplementedError()

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        """Computes the new teacher weights from the internal state.

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            New teacher weights
        """
        raise NotImplementedError()

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        """Called once before training. Useful to setup the initial internal state.

        Parameters
        ----------
        initial_model_w : OrderedDict[str, torch.Tensor]
            Current (initial) model weights.
        """
        raise NotImplementedError()

    def update_teacher_and_cache(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        # will indicate if teacher cache needs to be updated
        cache_dirty = self._teacher_update_state(progress, model)

        # If a weight has changed, compute updated teacher weights, cache it, and assign it
        if cache_dirty:
            try:
                new_teacher_w = self._compute_teacher_weights()
                self.cache = new_teacher_w
                model.task.teacher.load_state_dict(new_teacher_w)
            except AttributeError as err:
                print(f"TeacherUpdate callback can't be applied on this model : {err}")

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

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Only initialize once
        if self.initialized:
            return
        self.initialized = True

        initial_teacher_w = pl_module.task.teacher.state_dict()
        self._setup_initial(initial_teacher_w)
        self.cache = initial_teacher_w

    @staticmethod
    def get_averaged_weights(
        weights: List[OrderedDict[str, torch.Tensor]]
    ) -> OrderedDict[str, torch.Tensor]:
        return {
            k: torch.mean(torch.stack([w[k] for w in weights]), dim=0)
            for k in weights[0]
        }


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


class TeacherQueueUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        average_of: int = 1,
        update_interval: int = 1,
    ):
        """The teacher is the average of a queue of the last states of the student.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When should the update happen, by default "epoch"
        average_of : int, optional
            Queue max length, by default 1
        update_interval : int, optional
            Update will happen every 'update_interval' epochs/batches, by default 1
        """
        super().__init__(when=when)

        self.average_of = average_of
        self.update_interval = update_interval
        self.weights_queue: List[OrderedDict[str, torch.Tensor]] = None
        pass

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.weights_queue = []

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return TeacherUpdate.get_averaged_weights(self.weights_queue)

    def _teacher_update_state(self, progress: int, model: pl.LightningModule) -> bool:
        if progress % self.update_interval != 0:
            return False

        # Get new teacher "candidate" (from decayed weights) and enqueue it in the teacher history
        teacher_candidate_w = model.state_dict()
        self.enqueue_to_team(teacher_candidate_w)
        return True

    def enqueue_to_team(self, teacher: OrderedDict[str, torch.Tensor]):
        self.weights_queue.append(teacher)
        if len(self.weights_queue) > self.average_of:
            self.weights_queue.pop(0)


class TeacherCopyUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
    ):
        """Instant weights copy.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When should the update happen, by default "epoch"
        update_interval : int, optional
            Update will happen every 'update_interval' epochs/batches, by default 1
        """

        super().__init__(when=when)

        self.update_interval = update_interval
        self.last_weights = None

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.last_weights = initial_model_w

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.last_weights

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        if progress % self.update_interval != 0:
            return False

        self.last_weights = current_model.state_dict()
        return True


class TeacherEmaUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
        update_rate: float = 0.999,
    ):
        """Instant weights copy.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When should the update happen, by default "epoch"
        update_interval : int, optional
            Update will happen every 'update_interval' epochs/batches, by default 1
        update_rate : float, optional
            How much to keep of the old weights each update. 0=instant copy, 1=never update weights.
        """

        super().__init__(when=when)

        if update_rate < 0.0 or update_rate > 1.0:
            raise ValueError(
                f"Illegal update rate value ({update_rate}), it should be in [0.0,1.0]"
            )
        if update_rate == 0.0:
            warnings.warn(
                "You are using an EMA update with 0.0 update rate, consider using a TeacherCopyUpdate instead."
            )

        self.update_interval = update_interval
        self.update_rate = update_rate
        self.teacher_weights = None

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.teacher_weights = initial_model_w

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.teacher_weights

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        if progress % self.update_interval != 0:
            return False

        self.teacher_weights = get_decayed_weights(
            teacher_w=self.teacher_weights,
            student_w=current_model.state_dict(),
            tau=self.update_rate,
        )
        return True


class TeacherTeamUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"],
        update_interval: List[int],
        weight_update_rate: List[float],
    ):
        """TeacherTeamUpdate callback.
        Keeps in memory a "team" of teachers weights that gets updated at different interval and at a different rate (EMA).
        The final teacher is the average of all these weights.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When the update is applied, by default 'epoch'
        update_interval : List[int]
            How frequently the teacher(s) are updated.
        weight_update_rate : List[float]
            How much of the old weights to keep: 0.0 means instant update, 1.0 means never update.
        """
        super().__init__(when=when)

        if len(update_interval) != len(weight_update_rate):
            raise ValueError(
                "update_interval should be the same size as weight_update_rate"
            )

        self.update_interval = update_interval
        self.weight_update_rate = weight_update_rate

        # The weights of the team of teachers that will be averaged
        self.team_weights: List[OrderedDict[str, torch.Tensor]] = []

    def get_update_rate(self, index) -> float:
        return self.weight_update_rate[index]

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.team_weights = [initial_model_w] * len(self.update_interval)

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return TeacherUpdate.get_averaged_weights(self.team_weights)

    def _teacher_update_state(self, progress: int, model: pl.LightningModule) -> bool:
        cache_dirty = False
        # Check for each member of the ensemble if it needs to be updated
        for i in range(len(self.update_interval)):
            interval = self.update_interval[i]
            if interval == 0 or progress % interval != 0:
                continue

            new_teacher_i_w = get_decayed_weights(
                teacher_w=self.team_weights[i],
                student_w=model.state_dict(),
                tau=self.get_update_rate(i),
            )
            self.team_weights[i] = new_teacher_i_w
            cache_dirty = True
        return cache_dirty
