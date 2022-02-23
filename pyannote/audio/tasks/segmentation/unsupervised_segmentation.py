from typing import List, Optional, Text, Tuple, Union

import numpy as np
import torch
from pyannote.core import Segment
from pyannote.database import Protocol
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
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
        )

        self.m0 = model
        self.fake_in_train = fake_in_train
        self.fake_in_val = fake_in_val
        self.augmentation_model = augmentation_model

        self.m0.eval()

    def get_model_output(self, model: Model, waveforms: torch.Tensor):
        result = None
        # try inference mode ?
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            result = model(
                waveforms=waveforms
            ).detach()  # detach is necessary to avoid memory leaks
            result = torch.round(result).type(torch.int8)
        return result

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with m0 if they are not provided
        if "y" not in collated_batch:
            m0_input = collated_batch["X"]
            if self.augmentation_model is not None:
                m0_input = self.augmentation_model(
                    collated_batch["X"], sample_rate=self.model.hparams.sample_rate
                )
            collated_batch["y"] = self.get_model_output(self.m0, m0_input)

        if self.augmentation is not None:
            collated_batch["X"] = self.augmentation(
                collated_batch["X"], sample_rate=self.model.hparams.sample_rate
            )
        return collated_batch

    def collate_fn_val(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with m0 if they are not provided
        if "y" not in collated_batch:
            m0_input = collated_batch["X"]
            collated_batch["y"] = self.get_model_output(self.m0, m0_input)

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

        use_annotations = (stage == "train" and not self.fake_in_train) or (
            stage == "val" and not self.fake_in_val
        )
        sample = super().prepare_chunk(
            file, chunk, duration=duration, stage=stage, use_annotations=use_annotations
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
