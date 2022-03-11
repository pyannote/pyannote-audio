from typing import Tuple

import torch
from torch import Tensor

from pyannote.audio.utils.permutation import permutate


def _check_valid_tensors(preds: Tensor, target: Tensor):
    """Check both tensors have shape (NUM_BATCH, NUM_CLASSES, NUM_FRAMES) with the same NUM_BATCH and NUM_FRAMES."""
    if len(preds.shape) != 3 or len(target.shape) != 3:
        msg = f"Wrong shape ({tuple(target.shape)} or {tuple(preds.shape)}), expected (NUM_BATCH, NUM_CLASSES, NUM_FRAMES)."
        raise ValueError(msg)

    batch_size, _, num_samples = target.shape
    batch_size_, _, num_samples_ = preds.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(target.shape)} vs. {tuple(preds.shape)}. Both tensors should have the same NUM_BATCH and NUM_FRAMES."
        raise ValueError(msg)


def _der_update(
    preds: Tensor, target: Tensor, threshold: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the false alarm, missed detection, confusion and total values.

    Parameters
    ----------
    preds : torch.Tensor
        preds torch.tensor of shape (B,C,F)
    target : torch.Tensor
        preds torch.tensor of shape (B,C,F) (must only contain 0s and 1s)
    threshold : float
        threshold to discretize preds

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tensors with 1 item for false alarm, missed detection, confusion, and total
    """

    _check_valid_tensors(preds, target)

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


def _der_compute(
    false_alarm: Tensor, missed_detection: Tensor, confusion: Tensor, total: Tensor
) -> Tensor:
    return (false_alarm + missed_detection + confusion) / total


def der(preds: Tensor, target: Tensor, threshold: float = 0.5):
    fa, md, conf, total = _der_update(preds, target)
    return _der_compute(fa, md, conf, total)
