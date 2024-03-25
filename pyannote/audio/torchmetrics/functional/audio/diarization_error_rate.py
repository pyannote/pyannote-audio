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
import torch.nn.functional as F

from pyannote.audio.utils.permutation import permutate


def _der_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = 0.5,
    reduce: str = "batch",
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
    reduce : {'batch', 'chunk', 'frame'}, optional
        Reduction method. Defaults to 'batch'.

    Returns
    -------
    false_alarm : torch.Tensor
    missed_detection : torch.Tensor
    speaker_confusion : torch.Tensor
        If `reduce` is 'batch', returns (num_thresholds, )-shaped tensors.
        If `reduce` is 'chunk', returns (batch_size, num_thresholds)-shaped tensors.
        If `reduce` is 'frame', returns (batch_size, num_frames, num_thresholds)-shaped tensors.
        In case `threshold` is a float, the last dimension is removed from the output tensors.
    speech_total : (...,)-shaped torch.Tensor torch.Tensor
        If `reduce` is 'batch', returns a scalar.
        If `reduce` is 'chunk', returns (batch_size,)-shaped tensor.
        If `reduce` is 'frame', returns (batch_size, num_frames)-shaped tensor.
    """

    prd_batch_size, prd_num_speakers, prd_num_frames = preds.shape
    tgt_batch_size, tgt_num_speakers, tgt_num_frames = target.shape

    if prd_batch_size != tgt_batch_size:
        raise ValueError(f"Batch size mismatch: {prd_batch_size} != {tgt_batch_size}.")

    if prd_num_frames != tgt_num_frames:
        raise ValueError(
            f"Number of frames mismatch: {prd_num_frames} != {tgt_num_frames}."
        )

    # pad number of speakers if necessary
    if prd_num_speakers > tgt_num_speakers:
        target = F.pad(target, (0, 0, 0, prd_num_speakers - tgt_num_speakers))
    elif prd_num_speakers < tgt_num_speakers:
        preds = F.pad(preds, (0, 0, 0, tgt_num_speakers - prd_num_speakers))

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

    speech_total = 1.0 * torch.sum(target, 1)
    # (batch_size, num_frames)

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

    if reduce == "frame":
        if scalar_threshold:
            return (
                false_alarm[:, :, 0],
                missed_detection[:, :, 0],
                speaker_confusion[:, :, 0],
                speech_total,
            )
        return false_alarm, missed_detection, speaker_confusion, torch.sum(target, 1)

    speech_total = torch.sum(speech_total, 1)
    # (batch_size, )
    false_alarm = torch.sum(false_alarm, 1)
    missed_detection = torch.sum(missed_detection, 1)
    speaker_confusion = torch.sum(speaker_confusion, 1)
    # (batch_size, num_thresholds)

    if reduce == "chunk":
        if scalar_threshold:
            return (
                false_alarm[:, 0],
                missed_detection[:, 0],
                speaker_confusion[:, 0],
                speech_total,
            )
        return false_alarm, missed_detection, speaker_confusion, speech_total

    speech_total = torch.sum(speech_total, 0)
    # scalar
    false_alarm = torch.sum(false_alarm, 0)
    missed_detection = torch.sum(missed_detection, 0)
    speaker_confusion = torch.sum(speaker_confusion, 0)
    # (num_thresholds, )

    if scalar_threshold:
        return (
            false_alarm[0],
            missed_detection[0],
            speaker_confusion[0],
            speech_total,
        )

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
    reduce: str = "batch",
    return_components: bool = False,
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
    reduce : {'batch', 'chunk', 'frame'}, optional
        Reduction method. Defaults to 'batch'.
    return_components : bool, optional
        Return diarization error rate components as an additional tuple.
        Defaults to False.


    Returns
    -------
    der : torch.Tensor
        If `reduce` is 'batch', returns (num_thresholds, )-shaped tensors.
        If `reduce` is 'chunk', returns (batch_size, num_thresholds)-shaped tensors.
        If `reduce` is 'frame', returns (batch_size, num_frames, num_thresholds)-shaped tensors.
        In case `threshold` is a float, the last dimension is removed from the output tensors.
    components : (false_alarm, missed_detection, speaker_confusion, speech_total) tuple, optional
        Same shape as `der`. Only returned when `return_components` is True.

    """
    false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
        preds, target, threshold=threshold, reduce=reduce
    )

    der = _der_compute(false_alarm, missed_detection, speaker_confusion, speech_total)
    if return_components:
        return der, (false_alarm, missed_detection, speaker_confusion, speech_total)
    return der


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
