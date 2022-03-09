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


def der_check_tensors(preds: torch.Tensor, target: torch.Tensor):
    """Check for correct tensors shape.

    Parameters
    ----------
    preds : torch.Tensor
        Preds with shape (B,C,F)
    target : torch.Tensor
        Target with the shape (B,C,F)

    Raises
    ------
    ValueError
        Raised when the tensors have different shapes or shape of length different than 3
    """
    if len(preds.shape) < 3 or len(target.shape) < 3:
        msg = f"Wrong shape ({tuple(target.shape)} or {tuple(preds.shape)}), expected (NUM_BATCH, NUM_CLASSES, NUM_FRAMES)."
        raise ValueError(msg)

    batch_size, num_classes, num_samples = target.shape
    batch_size_, num_classes_, num_samples_ = preds.shape
    if (
        batch_size != batch_size_
        or num_classes != num_classes_
        or num_samples != num_samples_
    ):
        msg = f"Shape mismatch: {tuple(target.shape)} vs. {tuple(preds.shape)}."
        raise ValueError(msg)


def compute_der_values(
    preds: torch.Tensor, target: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the false alarm, missed detection, confusion and total values.

    Parameters
    ----------
    preds : torch.Tensor
        preds tensor of shape (B,C,F)
    target : torch.Tensor
        preds tensor of shape (B,C,F) (must only contain 0s and 1s)
    threshold : float
        threshold to discretize preds

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tensors with 1 item for false alarm, missed detection, confusion, and total
    """

    preds_bin = (preds > threshold).float()

    # convert to/from pyannote's tensor ordering (batch,frames,class) (instead of (batch,class,frames))
    hypothesis, _ = permutate(
        torch.transpose(target, 1, 2), torch.transpose(preds_bin, 1, 2)
    )
    hypothesis = torch.transpose(hypothesis, 1, 2)

    detection_error = torch.sum(hypothesis, 1) - torch.sum(target, 1)
    false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
    missed_detection = torch.maximum(
        -detection_error, torch.zeros_like(detection_error)
    )

    confusion = torch.sum((hypothesis != target) * hypothesis, 1) - false_alarm

    false_alarm = torch.sum(false_alarm)
    missed_detection = torch.sum(missed_detection)
    confusion = torch.sum(confusion)
    total = 1.0 * torch.sum(target)

    return false_alarm, missed_detection, confusion, total


class DER(Metric):
    """
    Compute Diarization Error Rate on discretized annotations.

    Expects preds and target tensors of the shape (NUM_BATCH, NUM_CLASSES, NUM_FRAMES) in its update.

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
    ):
        der_check_tensors(preds, target)

        false_alarm, missed_detection, confusion, total = compute_der_values(
            preds, target, self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.confusion += confusion
        self.total += total

    def compute(self):
        return (self.false_alarm + self.missed_detection + self.confusion) / self.total


class ConfusionMetric(DER):
    def compute(self):
        return self.confusion / self.total


class FalseAlarmMetric(DER):
    def compute(self):
        return self.false_alarm / self.total


class MissedDetectionMetric(DER):
    def compute(self):
        return self.missed_detection / self.total


class AUDER(Metric):
    """Area Under the Diarization Error Rate.
    Approximates the area under the curve of the DER when varying its threshold value.

    Expects preds and target tensors of the shape (NUM_BATCH, NUM_CLASSES, NUM_FRAMES) in its update.

    Note that this is only a reliable metric if num_frames == the total number of frames of the diarized audio.
    """

    def __init__(
        self, steps=31, threshold_min=0.0, threshold_max=1.0, force_unit_area=True
    ):
        super().__init__()

        if threshold_max < threshold_min:
            raise ValueError(
                f"Illegal value : threshold_max ({threshold_max}) < threshold_min ({threshold_min})"
            )

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.force_unit_area = force_unit_area
        self.steps = steps
        self.linspace = np.linspace(threshold_min, threshold_max, self.steps)
        # dx used for area computation. If we want an area in [0,1], fake it.
        self.area_dx = (threshold_max - threshold_min) if not force_unit_area else 1.0
        self.area_dx /= self.steps - 1

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
    ):
        der_check_tensors(preds, target)

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
