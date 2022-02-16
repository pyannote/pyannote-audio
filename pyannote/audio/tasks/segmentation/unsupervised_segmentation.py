from typing import Text, Tuple, Union

from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.tasks import Segmentation


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        # unsupervised params
        model: Model,
        # other params
        protocol: Protocol,
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
