# The MIT License (MIT)
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple

import numpy as np
import torch
from torchmetrics import Metric

from pyannote.audio.utils.permutation import permutate


def der_try_reshape(
    preds: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    num_frames: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tries to reshape preds and target into their expected shape (batch_size, num_frames, num_classes).
    Raises an error if it can't do it.

    Parameters
    ----------
    preds : torch.Tensor
        The update method preds parameter
    target : torch.Tensor
        The update method target parameter
    batch_size : int
        Batch size.
    num_frames : int
        Number of frames.
    num_classes : int
        Number of classes.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The tuple (preds, target) in the expected shape.

    Raises
    ------
    ValueError
        Raised if there isn't enough information to reshape preds and target.
    """
    if len(preds.shape) == 3 and len(target.shape) == 3:
        return preds, target

    # Preds or target do not have the correct size
    if batch_size == -1 or num_frames == -1 or num_classes == -1:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes), pass batch_size, num_frames and num_classes as parameters to the update function if needed."
        raise ValueError(msg)

    # Preds or target do not have the correct size AND can be resized
    preds = preds.reshape(batch_size, num_frames, num_classes)
    target = target.reshape(batch_size, num_frames, num_classes)
    return preds, target


def der_dim_check(preds: torch.Tensor, target: torch.Tensor):
    batch_size, num_samples, num_classes_1 = target.shape
    batch_size_, num_samples_, num_classes_2 = preds.shape
    if (
        batch_size != batch_size_
        or num_samples != num_samples_
        or num_classes_1 != num_classes_2
    ):
        msg = f"Shape mismatch: {tuple(target.shape)} vs. {tuple(preds.shape)}."
        raise ValueError(msg)


def compute_der_values(preds: torch.Tensor, target: torch.Tensor, threshold: float):
    preds_bin = (preds > threshold).float()

    hypothesis, _ = permutate(target, preds_bin)

    detection_error = torch.sum(hypothesis, 2) - torch.sum(target, 2)
    false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
    missed_detection = torch.maximum(
        -detection_error, torch.zeros_like(detection_error)
    )

    confusion = torch.sum((hypothesis != target) * hypothesis, 2) - false_alarm

    false_alarm = torch.sum(false_alarm)
    missed_detection = torch.sum(missed_detection)
    confusion = torch.sum(confusion)
    total = 1.0 * torch.sum(target)
    return false_alarm, missed_detection, confusion, total


class DER(Metric):
    """
    Compute diarization error rate on discretized annotations with torchmetrics

    Note that this is only a reliable metric if num_frames == the total number of frames of the diarized audio.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.threshold = threshold

        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("confusion", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        batch_size: int = -1,
        num_frames: int = -1,
        num_classes: int = -1,
    ):
        preds, target = der_try_reshape(
            preds, target, batch_size, num_frames, num_classes
        )
        der_dim_check(preds, target)

        false_alarm, missed_detection, confusion, total = compute_der_values(
            preds, target, self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.confusion += confusion
        self.total += total

    def compute(self):
        return (self.false_alarm + self.missed_detection + self.confusion) / self.total


class AUDER(Metric):
    """Area Under the Diarization Error Rate.
    Approximates the area under the curve of the DER when varying its threshold value.

    Note that this is only a reliable metric if num_frames == the total number of frames of the diarized audio.
    """

    def __init__(
        self, dx=0.0333, threshold_min=0.0, threshold_max=1.0, force_unit_area=True
    ):
        super().__init__()

        if threshold_max < threshold_min:
            raise ValueError(
                f"Illegal value : threshold_max ({threshold_max}) < threshold_min ({threshold_min})"
            )

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.force_unit_area = force_unit_area
        self.steps = int(threshold_max - threshold_min) / dx + 1
        # dx used for area computation. If we want an area in [0,1], fake it.
        self.area_dx = dx if not force_unit_area else 1.0 / (self.steps - 1)
        self.linspace = np.linspace(threshold_min, threshold_max, self.steps)

        self.add_state(
            "false_alarm",
            default=torch.zeros(self.steps, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "missed_detection",
            torch.zeros(self.steps, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "confusion",
            torch.zeros(self.steps, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", torch.zeros(self.steps, dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        batch_size: int = -1,
        num_frames: int = -1,
        num_classes: int = -1,
    ):
        preds, target = der_try_reshape(
            preds, target, batch_size, num_frames, num_classes
        )
        der_dim_check(preds, target)

        for i in range(self.steps):
            threshold = self.linspace[i]

            false_alarm, missed_detection, confusion, total = compute_der_values(
                preds, target, threshold
            )
            self.false_alarm[i] += false_alarm
            self.missed_detection[i] += missed_detection
            self.confusion[i] += confusion
            self.total[i] += total

    def compute(self):
        ders = (self.false_alarm + self.missed_detection + self.confusion) / self.total
        area = torch.trapezoid(ders, dx=self.area_dx)
        return area
