# MIT License
#
# Copyright (c) 2022- CNRS
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

from typing import Optional

import torch
from torchmetrics import Metric

from pyannote.audio.torchmetrics.functional.audio.diarization_error_rate import (
    _der_compute,
    _der_update,
)


class DiarizationErrorRate(Metric):
    """Diarization error rate

    Parameters
    ----------
    threshold : float, optional
        Threshold used to binarize predictions. Defaults to 0.5.

    Notes
    -----
    While pyannote.audio conventions is to store speaker activations with
    (batch_size, num_frames, num_speakers)-shaped tensors, this torchmetrics metric
    expects them to be shaped as (batch_size, num_speakers, num_frames) tensors.
    """

    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.threshold = threshold

        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "speaker_confusion", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Compute and accumulate components of diarization error rate

        Parameters
        ----------
        preds : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped continuous predictions.
        target : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.

        Returns
        -------
        false_alarm : torch.Tensor
        missed_detection : torch.Tensor
        speaker_confusion : torch.Tensor
        speech_total : torch.Tensor
            Diarization error rate components accumulated over the whole batch.
        """

        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.speaker_confusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        return _der_compute(
            self.false_alarm,
            self.missed_detection,
            self.speaker_confusion,
            self.speech_total,
        )


class SpeakerConfusionRate(DiarizationErrorRate):
    def compute(self):
        return self.speaker_confusion / (self.speech_total + 1e-8)


class FalseAlarmRate(DiarizationErrorRate):
    def compute(self):
        return self.false_alarm / (self.speech_total + 1e-8)


class MissedDetectionRate(DiarizationErrorRate):
    def compute(self):
        return self.missed_detection / (self.speech_total + 1e-8)


class OptimalDiarizationErrorRate(Metric):
    """Optiml Diarization error rate

    Parameters
    ----------
    thresholds : torch.Tensor, optional
        Thresholds used to binarize predictions.
        Defaults to torch.linspace(0.0, 1.0, 51)

    Notes
    -----
    While pyannote.audio conventions is to store speaker activations with
    (batch_size, num_frames, num_speakers)-shaped tensors, this torchmetrics metric
    expects them to be shaped as (batch_size, num_speakers, num_frames) tensors.
    """

    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: Optional[torch.Tensor] = None):
        super().__init__()

        threshold = threshold or torch.linspace(0.0, 1.0, 51)
        self.add_state("threshold", default=threshold, dist_reduce_fx="mean")
        (num_thresholds,) = threshold.shape

        # note that CamelCase is used to indicate that those states contain values for multiple thresholds
        # this is for torchmetrics to know that these states are different from those of DiarizationErrorRate
        # for which only one threshold is used.

        self.add_state(
            "FalseAlarm",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "MissedDetection",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "SpeakerConfusion",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Compute and accumulate components of diarization error rate

        Parameters
        ----------
        preds : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped continuous predictions.
        target : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.

        Returns
        -------
        false_alarm : torch.Tensor
        missed_detection : torch.Tensor
        speaker_confusion : torch.Tensor
        speech_total : torch.Tensor
            Diarization error rate components accumulated over the whole batch.
        """

        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.FalseAlarm += false_alarm
        self.MissedDetection += missed_detection
        self.SpeakerConfusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        opt_der, _ = torch.min(der, dim=0)

        return opt_der


class OptimalDiarizationErrorRateThreshold(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        opt_threshold = self.threshold[opt_threshold_idx]

        return opt_threshold


class OptimalSpeakerConfusionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.SpeakerConfusion[opt_threshold_idx] / (self.speech_total + 1e-8)


class OptimalFalseAlarmRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.FalseAlarm[opt_threshold_idx] / (self.speech_total + 1e-8)


class OptimalMissedDetectionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.MissedDetection[opt_threshold_idx] / (self.speech_total + 1e-8)
