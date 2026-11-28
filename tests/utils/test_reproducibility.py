# MIT License
#
# Copyright (c) 2026- CNRS
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

"""Tests for pyannote.audio.utils.reproducibility.

See https://github.com/pyannote/pyannote-audio/issues/2108:
`fix_reproducibility` must only disable TensorFloat-32 (TF32) for the
duration of pyannote inference and restore the previous settings on exit,
instead of turning TF32 off for the whole process.
"""

import warnings

import pytest
import torch

from pyannote.audio.utils.reproducibility import (
    ReproducibilityWarning,
    fix_reproducibility,
)


def _set_tf32(matmul: bool, cudnn: bool):
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def _get_tf32():
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def test_fix_reproducibility_restores_tf32_settings():
    """TF32 flags are off inside the context and restored afterwards."""
    _set_tf32(True, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fix_reproducibility(torch.device("cuda")):
            assert _get_tf32() == (False, False)
    assert _get_tf32() == (True, True)


def test_fix_reproducibility_restores_mixed_tf32_settings():
    """Each flag is restored to its own previous value, not just to True."""
    _set_tf32(True, False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fix_reproducibility(torch.device("cuda")):
            assert _get_tf32() == (False, False)
    assert _get_tf32() == (True, False)


def test_fix_reproducibility_restores_tf32_settings_on_exception():
    """Flags are restored even when inference raises."""
    _set_tf32(True, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            with fix_reproducibility(torch.device("cuda")):
                assert _get_tf32() == (False, False)
                raise RuntimeError("boom")
    assert _get_tf32() == (True, True)


def test_fix_reproducibility_warns_that_settings_are_restored():
    """The warning no longer tells users to re-enable TF32 themselves."""
    _set_tf32(True, True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fix_reproducibility(torch.device("cuda")):
            pass
    assert len(caught) == 1
    assert issubclass(caught[0].category, ReproducibilityWarning)
    message = str(caught[0].message)
    assert "re-enabled" not in message
    assert "restored afterwards" in message
    assert _get_tf32() == (True, True)


def test_fix_reproducibility_is_a_noop_when_tf32_is_already_off():
    """No warning and no change when both flags are already False."""
    _set_tf32(False, False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fix_reproducibility(torch.device("cuda")):
            assert _get_tf32() == (False, False)
    assert caught == []
    assert _get_tf32() == (False, False)


def test_fix_reproducibility_is_a_noop_on_cpu():
    """TF32 settings are untouched (and no warning issued) on CPU."""
    _set_tf32(True, True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fix_reproducibility(torch.device("cpu")):
            assert _get_tf32() == (True, True)
    assert caught == []
    assert _get_tf32() == (True, True)
