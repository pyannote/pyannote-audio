import torch
from torch import Tensor
from torchmetrics import Metric

from pyannote.audio.torchmetrics.functional.audio.auder import (
    _auder_compute,
    _auder_update,
)


class AUDER(Metric):
    """Area Under the Diarization Error Rate.
    Approximates the area under the curve of the DER when varying its threshold value.

    Expects preds and target tensors of the shape (NUM_BATCH, NUM_CLASSES, NUM_FRAMES) in its update.

    Note that this is only a reliable metric if num_frames == the total number of frames of the diarized audio.
    """

    def __init__(self, steps=31, threshold_min=0.0, threshold_max=1.0, unit_area=True):
        super().__init__()

        if threshold_max < threshold_min:
            raise ValueError(
                f"Illegal value : threshold_max ({threshold_max}) < threshold_min ({threshold_min})"
            )

        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.unit_area = unit_area
        self.steps = steps

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
        preds: Tensor,
        target: Tensor,
    ):
        fa, md, conf, total = _auder_update(
            preds, target, self.steps, self.threshold_min, self.threshold_max
        )
        self.false_alarm += fa
        self.missed_detection += md
        self.confusion += conf
        self.total += total

    def compute(self):
        dx = (
            (self.threshold_max - self.threshold_min) if not self.unit_area else 1.0
        ) / (self.steps - 1)
        return _auder_compute(
            self.false_alarm, self.missed_detection, self.confusion, self.total, dx
        )
