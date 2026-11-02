#!/usr/bin/env python

# The MIT License (MIT)

# Copyright (c) 2023- CNRS

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


from pyannote.database.protocol import (
    CollectionProtocol,
    Protocol,
    SpeakerDiarizationProtocol,
    SpeakerVerificationProtocol,
)


def test_configured_protocols():
    from pathlib import Path

    from pyannote.database.registry import Registry

    registry = Registry()
    registry.load_database(Path(__file__).parent / "data/database.yml")
    assert "MyDatabase" in registry.databases

    database = registry.get_database("MyDatabase")
    tasks = database.get_tasks()
    assert "Collection" in tasks
    assert "Protocol" in tasks
    assert "SpeakerDiarization" in tasks
    assert "SpeakerVerification" in tasks

    assert "MyCollection" in database.get_protocols("Collection")
    assert "MyProtocol" in database.get_protocols("Protocol")
    assert "MySpeakerDiarization" in database.get_protocols("SpeakerDiarization")
    assert "MySpeakerVerification" in database.get_protocols("SpeakerVerification")

    collection = registry.get_protocol("MyDatabase.Collection.MyCollection")
    assert isinstance(collection, CollectionProtocol)

    protocol = registry.get_protocol("MyDatabase.Protocol.MyProtocol")
    assert isinstance(protocol, Protocol)

    speaker_diarization = registry.get_protocol(
        "MyDatabase.SpeakerDiarization.MySpeakerDiarization"
    )
    assert isinstance(speaker_diarization, SpeakerDiarizationProtocol)

    speaker_verification = registry.get_protocol(
        "MyDatabase.SpeakerVerification.MySpeakerVerification"
    )
    assert isinstance(speaker_verification, SpeakerVerificationProtocol)

    files = list(collection.files())
    assert len(files) == 2

    files = list(protocol.files())
    assert len(files) == 2

    files = list(speaker_diarization.files())
    assert len(files) == 2

    files = list(speaker_verification.files())
    assert len(files) == 2

    meta_protocol = registry.get_protocol("X.SpeakerDiarization.MyMetaProtocol")
    files = list(meta_protocol.train())
    assert len(files) == 2

    files = list(meta_protocol.development())
    assert len(files) == 4
