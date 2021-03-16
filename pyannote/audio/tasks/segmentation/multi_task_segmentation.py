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


from typing import Mapping, Text

import numpy as np
from pytorch_lightning.metrics.classification import FBeta
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.task import Problem, Task
from pyannote.audio.tasks import (
    OverlappedSpeechDetection,
    SpeakerChangeDetection,
    VoiceActivityDetection,
)
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database import Protocol


class MultiTaskSegmentation(SegmentationTaskMixin, Task):
    """Multi-task segmentation

    Multi-task training of segmentation tasks, including:
        - vad: voice activity detection
        - scd: speaker change detection
        - osd: overlapped speech detection

    Note that, when "osd" is one of the tasks considered, data augmentation is used
    to artificially increase the proportion of "overlap". This is achieved by
    generating chunks made out of the (weighted) sum of two random chunks.

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
    vad : bool, optional
        Add voice activity detection in the pool of tasks.
        Defaults to False.
    vad_params : dict, optional
        Additional vad-specific parameters. Has no effect when `vad` is False.
        See VoiceActivityDetection docstring for details and default value.
    scd : bool, optional
        Add speaker change detection to the pool of tasks.
        Defaults to False.
    scd_params : dict, optional
        Additional scd-specific parameters. Has no effect when `scd` is False.
        See SpeakerChangeDetection docstring for details and default value.
    osd : bool, optional
        Add overlapped speech detection to the pool of tasks.
        Defaults to False.
    osd_params : dict, optional
        Additional osd-specific parameters. Has no effect when `osd` is False.
        See OverlappedSpeechDetection docstring for details and default value.
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

    """

    ACRONYM = "xseg"

    OVERLAP_DEFAULTS = {"probability": 0.5, "snr_min": 0.0, "snr_max": 10.0}

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        overlap: dict = OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        vad: bool = False,
        vad_params: Mapping = None,
        scd: bool = False,
        scd_params: Mapping = None,
        osd: bool = False,
        osd_params: Mapping = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )

        self.vad = vad
        self.vad_params = dict() if vad_params is None else vad_params

        self.scd = scd
        self.scd_params = dict() if scd_params is None else scd_params

        self.osd = osd
        self.osd_params = dict() if osd_params is None else osd_params

        if sum([vad, scd, osd]) < 2:
            raise ValueError(
                "You must activate at least two tasks among 'vad', 'scd', and 'osd'."
            )

        self.tasks = dict()
        if self.vad:
            self.tasks["vad"] = VoiceActivityDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.vad_params,
            )
        if self.scd:
            self.tasks["scd"] = SpeakerChangeDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.scd_params,
            )
        if self.osd:
            self.tasks["osd"] = OverlappedSpeechDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.osd_params,
            )

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            self.specifications = {
                name: task.specifications for name, task in self.tasks.items()
            }

    def prepare_y(self, one_hot_y: np.ndarray):
        """Get multi-task targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is two or more active speakers at tth frame, 0 otherwise.
        """

        return {name: task.prepare_y(one_hot_y) for name, task in self.tasks.items()}

    def setup_validation_metric(self):

        self.val_fbeta = {
            task_name: FBeta(
                len(specifications.classes),
                beta=1.0,
                threshold=0.5,
                multilabel=(
                    specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION
                ),
                average="macro",
            )
            for task_name, specifications in self.specifications.items()
        }

    def validation_step(self, batch, batch_idx: int):
        """Compute areas under ROC curve

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        X, y = batch["X"], batch["y"]
        y_pred = self.model(X)

        val_fbeta = dict()

        for task_name in self.specifications:

            # move metric to model device
            self.val_fbeta[task_name].to(self.model.device)

            val_fbeta[task_name] = self.val_fbeta[task_name](
                y_pred[task_name][:, ::10].squeeze(), y[task_name][:, ::10].squeeze()
            )

            self.model.log(
                f"{task_name}@val_fbeta",
                val_fbeta[task_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.model.log(
            f"{self.ACRONYM}@val_fbeta",
            sum(val_fbeta.values()) / len(val_fbeta),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
