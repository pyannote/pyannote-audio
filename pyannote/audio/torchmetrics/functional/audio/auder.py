from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from pyannote.audio.torchmetrics.functional.audio.der import (
    _check_valid_tensors,
    _der_update,
)


def _auder_update(
    preds: Tensor,
    target: Tensor,
    steps: int,
    tmin: float,
    tmax: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    _check_valid_tensors(preds, target)

    false_alarm = torch.zeros(steps, dtype=torch.float, device=preds.device)
    missed_detection = torch.zeros(steps, dtype=torch.float, device=preds.device)
    confusion = torch.zeros(steps, dtype=torch.float, device=preds.device)
    total = torch.zeros(steps, dtype=torch.float, device=preds.device)

    linspace = np.linspace(tmin, tmax, steps)
    for i in range(steps):
        threshold = linspace[i]

        der_fa, der_md, der_conf, der_total = _der_update(preds, target, threshold)
        false_alarm[i] += der_fa
        missed_detection[i] += der_md
        confusion[i] += der_conf
        total[i] += der_total
    return false_alarm, missed_detection, confusion, total


def _auder_compute(
    false_alarm: Tensor,
    missed_detection: Tensor,
    confusion: Tensor,
    total: Tensor,
    dx: float,
) -> Tensor:
    ders = (false_alarm + missed_detection + confusion) / total
    return torch.trapezoid(ders, dx=dx)


def auder(
    preds: Tensor,
    target: Tensor,
    steps: int,
    tmin: float,
    tmax: float,
    unit_area: bool,
):
    fa, md, conf, total = _auder_update(preds, target, steps, tmin, tmax)
    dx = ((tmax - tmin) if not unit_area else 1.0) / (steps - 1)
    return _auder_update(fa, md, conf, total, dx)
