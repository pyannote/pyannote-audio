# MIT License
#
# Copyright (c) 2023- CNRS
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

# Context: https://github.com/pyannote/pyannote-audio/issues/1370

import warnings

import torch


class ReproducibilityError(Exception):
    ...


class ReproducibilityWarning(UserWarning):
    ...


def raise_reproducibility(device: torch.device):
    if (device.type == "cuda") and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        raise ReproducibilityError(
            "Please disable TensorFloat-32 (TF32) by calling\n"
            "   >>> import torch\n"
            "   >>> torch.backends.cuda.matmul.allow_tf32 = False\n"
            "   >>> torch.backends.cudnn.allow_tf32 = False\n"
            "or you might face reproducibility issues and obtain lower accuracy.\n"
            "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details."
        )


def warn_reproducibility(device: torch.device):
    if (device.type == "cuda") and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        warnings.warn(
            ReproducibilityWarning(
                "Please disable TensorFloat-32 (TF32) by calling\n"
                "   >>> import torch\n"
                "   >>> torch.backends.cuda.matmul.allow_tf32 = False\n"
                "   >>> torch.backends.cudnn.allow_tf32 = False\n"
                "or you might face reproducibility issues and obtain lower accuracy.\n"
                "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details."
            )
        )


def fix_reproducibility(device: torch.device):
    if (device.type == "cuda") and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        warnings.warn(
            ReproducibilityWarning(
                "TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.\n"
                "It can be re-enabled by calling\n"
                "   >>> import torch\n"
                "   >>> torch.backends.cuda.matmul.allow_tf32 = True\n"
                "   >>> torch.backends.cudnn.allow_tf32 = True\n"
                "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.\n"
            )
        )
