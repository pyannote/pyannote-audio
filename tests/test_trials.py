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
# Hervé BREDIN - http://herve.niderb.fr
# Alexis PLAQUET


from pathlib import Path

from pyannote.core import Annotation, Timeline

from pyannote.database.registry import Registry


def test_verification_trials():
    registry = Registry()
    registry.load_database(Path(__file__).parent / "data/database.yml")
    protocol = registry.get_protocol(
        "MyDatabase.SpeakerVerification.MySpeakerVerification"
    )
    trials = list(protocol.train_trial())
    assert len(trials) == 1
    trial = trials[0]
    assert not trial["reference"]
    for key, uri in (("file1", "filename1"), ("file2", "filename2")):
        file = trial[key]
        assert file["uri"] == uri
        assert isinstance(file["annotation"], Annotation)
        assert isinstance(file["annotated"], Timeline)
        assert file["try_with"].duration() > 0
