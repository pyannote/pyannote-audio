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


from numbers import Number
from typing import Optional, Tuple, Union

import torch

from pyannote.audio.utils.permutation import permutate


def _der_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute components of diarization error rate

    Parameters
    ----------
    preds : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped continuous predictions.
    target : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.
    threshold : float or torch.Tensor, optional
        Threshold(s) used to binarize predictions. Defaults to 0.5.

    Returns
    -------
    false_alarm : (num_thresholds, )-shaped torch.Tensor
    missed_detection : (num_thresholds, )-shaped torch.Tensor
    speaker_confusion : (num_thresholds, )-shaped torch.Tensor
    speech_total : torch.Tensor
        Diarization error rate components accumulated over the whole batch.
    """

    # make threshold a (num_thresholds,) tensor
    scalar_threshold = isinstance(threshold, Number)
    if scalar_threshold:
        threshold = torch.tensor([threshold], dtype=preds.dtype, device=preds.device)

    # find the optimal mapping between target and (soft) predictions
    permutated_preds, _ = permutate(
        torch.transpose(target, 1, 2), torch.transpose(preds, 1, 2)
    )
    permutated_preds = torch.transpose(permutated_preds, 1, 2)
    # (batch_size, num_speakers, num_frames)

    # turn continuous [0, 1] predictions into binary {0, 1} decisions
    hypothesis = (permutated_preds.unsqueeze(-1) > threshold).float()
    # (batch_size, num_speakers, num_frames, num_thresholds)

    target = target.unsqueeze(-1)
    # (batch_size, num_speakers, num_frames, 1)

    detection_error = torch.sum(hypothesis, 1) - torch.sum(target, 1)
    # (batch_size, num_frames, num_thresholds)

    false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
    # (batch_size, num_frames, num_thresholds)

    missed_detection = torch.maximum(
        -detection_error, torch.zeros_like(detection_error)
    )
    # (batch_size, num_frames, num_thresholds)

    speaker_confusion = torch.sum((hypothesis != target) * hypothesis, 1) - false_alarm
    # (batch_size, num_frames, num_thresholds)

    false_alarm = torch.sum(torch.sum(false_alarm, 1), 0)
    missed_detection = torch.sum(torch.sum(missed_detection, 1), 0)
    speaker_confusion = torch.sum(torch.sum(speaker_confusion, 1), 0)
    # (num_thresholds, )

    speech_total = 1.0 * torch.sum(target)

    if scalar_threshold:
        false_alarm = false_alarm[0]
        missed_detection = missed_detection[0]
        speaker_confusion = speaker_confusion[0]

    return false_alarm, missed_detection, speaker_confusion, speech_total


def _der_compute(
    false_alarm: torch.Tensor,
    missed_detection: torch.Tensor,
    speaker_confusion: torch.Tensor,
    speech_total: torch.Tensor,
) -> torch.Tensor:
    """Compute diarization error rate from its components

    Parameters
    ----------
    false_alarm : (num_thresholds, )-shaped torch.Tensor
    missed_detection : (num_thresholds, )-shaped torch.Tensor
    speaker_confusion : (num_thresholds, )-shaped torch.Tensor
    speech_total : torch.Tensor
        Diarization error rate components, in number of frames.

    Returns
    -------
    der : (num_thresholds, )-shaped torch.Tensor
        Diarization error rate.
    """

    return (false_alarm + missed_detection + speaker_confusion) / (speech_total + 1e-8)


def diarization_error_rate(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = 0.5,
) -> torch.Tensor:
    """Compute diarization error rate

    Parameters
    ----------
    preds : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped continuous predictions.
    target : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.
    threshold : float or torch.Tensor, optional
        Threshold(s) used to binarize predictions. Defaults to 0.5.

    Returns
    -------
    der : (num_thresholds, )-shaped torch.Tensor
        Aggregated diarization error rate
    """
    false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
        preds, target, threshold=threshold
    )
    return _der_compute(false_alarm, missed_detection, speaker_confusion, speech_total)


def optimal_diarization_error_rate(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute optimal diarization error rate

    Parameters
    ----------
    preds : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped continuous predictions.
    target : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.
    thresholds : torch.Tensor, optional
        Thresholds used to binarize predictions.
        Defaults to torch.linspace(0.0, 1.0, 51)

    Returns
    -------
    opt_der : torch.Tensor
    opt_threshold : torch.Tensor
        Optimal threshold and corresponding diarization error rate.
    """

    threshold = threshold or torch.linspace(0.0, 1.0, 51, device=preds.device)
    der = diarization_error_rate(preds, target, threshold=threshold)
    opt_der, opt_threshold_idx = torch.min(der, dim=0)
    return opt_der, threshold[opt_threshold_idx]
