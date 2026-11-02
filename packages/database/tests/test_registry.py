#!/usr/bin/env python

# The MIT License (MIT)

# Copyright (c) 2023- CNRS
# Copyright (c) 2023- Université Paul Sabatier

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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Alexis PLAQUET
# Hervé BREDIN - http://herve.niderb.fr

import warnings

import pytest

from pyannote.database.registry import LoadingMode, _merge_protocols_inplace


def test_override_merging_disjoint():
    protocols1 = {
        ("Task1", "Protocol1"): None,
    }
    protocols2 = {
        ("OtherTask", "Protocol1"): 42,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # expect no warning
        _merge_protocols_inplace(protocols1, protocols2, LoadingMode.KEEP, "", "")

    assert (
        "Task1",
        "Protocol1",
    ) in protocols1
    assert (
        "OtherTask",
        "Protocol1",
    ) in protocols1
    assert len(protocols1) == 2


def test_override_merging_identical():

    protocols2 = {
        ("Task1", "Protocol1"): None,
    }  # the "old" protocols dict. KEEP override options will keep these entries.

    # Expect warning and protocols1 to become protocols2 (keep old value)
    protocols1 = {
        ("Task1", "Protocol1"): 42,
    }
    with pytest.warns(Warning):
        _merge_protocols_inplace(protocols1, protocols2, LoadingMode.KEEP, "", "")
        assert ("Task1", "Protocol1") in protocols1
        assert protocols1[("Task1", "Protocol1")] is None
        assert len(protocols1) == 1

    # Expect warning and protocols1 to keep its value (use new value)
    protocols1 = {
        ("Task1", "Protocol1"): 42,
    }
    with pytest.warns(Warning):
        _merge_protocols_inplace(protocols1, protocols2, LoadingMode.OVERRIDE, "", "")
        assert ("Task1", "Protocol1") in protocols1
        assert protocols1[("Task1", "Protocol1")] == 42
        assert len(protocols1) == 1
