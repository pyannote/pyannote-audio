from torch import Tensor, tensor
from torchmetrics import Metric

from pyannote.audio.torchmetrics.functional.audio.der import _der_compute, _der_update


class DER(Metric):
    """
    Compute Diarization Error Rate on discretized annotations.

    Expects preds and target tensors of the shape (NUM_BATCH, NUM_CLASSES, NUM_FRAMES) in its update.

    Note that this is only a reliable metric if num_frames == the total number of frames of the diarized audio.
    """

    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.threshold = threshold

        self.add_state("false_alarm", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("missed_detection", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("confusion", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ) -> None:
        false_alarm, missed_detection, confusion, total = _der_update(
            preds, target, self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.confusion += confusion
        self.total += total

    def compute(self):
        return _der_compute(
            self.false_alarm, self.missed_detection, self.confusion, self.total
        )


class ConfusionMetric(DER):
    def compute(self):
        return self.confusion / self.total


class FalseAlarmMetric(DER):
    def compute(self):
        return self.false_alarm / self.total


class MissedDetectionMetric(DER):
    def compute(self):
        return self.missed_detection / self.total
