# The MIT License (MIT)
#
# Copyright (c) 2017-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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

"""Resolve the device exposed by the active accelerator.

``torch.cuda`` is only one of the accelerators that ``torch`` knows about: XPU,
MPS and third-party backends (e.g. the Ascend NPU exposed by ``torch_npu``)
register themselves as well.  The helpers below ask ``torch.accelerator`` for
the accelerator that is actually installed instead of assuming CUDA, so that
those backends are used instead of silently falling back to CPU.
"""

import torch


def get_accelerator_device() -> torch.device:
    """Device of the active accelerator, or CPU when there is none."""

    try:
        if torch.accelerator.is_available():
            return torch.accelerator.current_accelerator()
    except (AttributeError, RuntimeError):
        # `torch.accelerator` is only available in torch >= 2.5.
        pass

    # `torch.accelerator` does not know about every backend (e.g. MPS on
    # builds that do not register it as an accelerator).
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_accelerator_device_type() -> str:
    """Device type of the active accelerator ("cuda", "npu", "cpu", ...)."""

    return get_accelerator_device().type


def get_accelerator_device_count() -> int:
    """Number of devices exposed by the active accelerator (0 when none)."""

    if get_accelerator_device_type() == "cpu":
        return 0

    try:
        return torch.accelerator.device_count()
    except (AttributeError, RuntimeError):
        return 0
