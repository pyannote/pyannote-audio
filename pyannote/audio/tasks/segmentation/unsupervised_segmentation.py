from typing import List, Text, Tuple, Union

import numpy as np
import torch
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from typing_extensions import Literal

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.tasks import Segmentation


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        model: Model,  # unsupervised param: model to use to generate truth
        protocol: Protocol,
        fake_in_train=True,  # generate fake truth in training mode
        fake_in_val=True,  # generate fake truth in val mode
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

    def get_truth_from_model(
        self, waveforms: torch.Tensor, model: Model = None
    ) -> Tuple[np.ndarray, List[Text]]:
        if model is None:
            model = self.m0

        y = self.m0(waveforms=waveforms)[0, :, :]

        # dirty float output to int output
        y = torch.round(y)
        y = y.type(torch.int8)

        # Generate dummy labels
        labels = [f"?{i}" for i in range(y.shape[-1])]  # ... ?

        return y.numpy(), labels

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

        sample = dict()

        if (stage == "train" and self.fake_in_train) or (
            stage == "val" and self.fake_in_val
        ):
            # X = "audio" crop
            sample["X"], _ = self.model.audio.crop(
                file,
                chunk,
                duration=self.duration if duration is None else duration,
            )
            # y and labels are generated from the model m0
            sample["y"], sample["labels"] = self.get_truth_from_model(
                sample["X"][None, :]
            )
        else:
            (
                sample["X"],
                sample["y"],
                sample["labels"],
            ) = self.get_x_y_labels_from_file_chunk(file, chunk, duration)

        # ==================================================================
        # additional metadata
        # ==================================================================

        for key, value in file.items():

            # those keys were already dealt with
            if key in ["audio", "annotation", "annotated"]:
                pass

            # replace text-like entries by their integer index
            elif isinstance(value, Text):
                try:
                    sample[key] = self._train_metadata[key].index(value)
                except ValueError as e:
                    if stage == "val":
                        sample[key] = -1
                    else:
                        raise e

            # crop score-like entries
            elif isinstance(value, SlidingWindowFeature):
                sample[key] = value.crop(chunk, fixed=duration, mode="center")

        return sample
