# MIT License
#
# Copyright (c) 2020-2025 CNRS
# Copyright (c) 2025- pyannoteAI
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

from typing import Optional

from pyannote.audio.core.io import Audio
from pyannote.core import Annotation
from pyannote.database import FileFinder, Protocol, get_annotated
from pyannote.database.protocol import SpeakerVerificationProtocol

get_duration = Audio(mono="downmix").get_duration


def check_protocol(protocol: Protocol) -> Protocol:
    """Check that protocol is suitable for training a model

        - does it provide a training set?
        - does it provide a validation set?
        - does it provide a way to access audio content?
        - does it provide a way to delimit annotated content?

    Returns
    -------
    fixed_protocol : Protocol
    checks: dict
        has_validation : bool
        has_scope : bool
        has_classes : bool

    Raises
    ------
    ValueError if protocol does not pass the check list and cannot be fixed.

    """

    # does protocol define a training set?
    try:
        file = next(protocol.train())
    except (AttributeError, NotImplementedError):
        msg = f"Protocol {protocol.name} does not define a training set."
        raise ValueError(msg)

    # does protocol provide audio keys?
    if "audio" not in file:
        if "waveform" in file:
            if "sample_rate" not in file:
                msg = f'Protocol {protocol.name} provides audio with "waveform" key but is missing a "sample_rate" key.'
                raise ValueError(msg)

        else:
            file_finder = FileFinder()
            try:
                _ = file_finder(file)

            except (KeyError, FileNotFoundError):
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files. "
                    f"See pyannote.database documentation on how to add an 'audio' preprocessor."
                )
                raise ValueError(msg)
            else:
                protocol.preprocessors["audio"] = file_finder
                msg = (
                    f"Protocol {protocol.name} does not provide the path to audio files: "
                    f"adding an 'audio' preprocessor for you. See pyannote.database documentation "
                    f"on how to do that yourself."
                )
                print(msg)

    if "annotated" not in file:
        if "duration" not in file:
            protocol.preprocessors["duration"] = get_duration

        protocol.preprocessors["annotated"] = get_annotated

        msg = (
            f"Protocol {protocol.name} does not provide the 'annotated' regions: "
            f"adding an 'annotated' preprocessor for you. See pyannote.database documentation "
            f"on how to do that yourself."
        )
        print(msg)

    has_scope = "scope" in file
    has_classes = "classes" in file

    # does protocol define a validation set?
    if isinstance(protocol, SpeakerVerificationProtocol):
        validation_method = "development_trial"
    else:
        validation_method = "development"

    try:
        _ = next(getattr(protocol, validation_method)())
    except (AttributeError, NotImplementedError):
        has_validation = False
    else:
        has_validation = True

    checks = {
        "has_validation": has_validation,
        "has_scope": has_scope,
        "has_classes": has_classes,
    }

    return protocol, checks


class FilterByNumberOfSpeakers:
    """Filter files based on the number of speakers

    Note
    ----
    Always returns True if `current_file` does not have an "annotation" key.

    """

    def __init__(
        self,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        from pyannote.audio.pipelines.utils.diarization import set_num_speakers

        self.num_speakers, self.min_speakers, self.max_speakers = set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def __call__(self, current_file: dict) -> bool:
        if "annotation" not in current_file:
            return True

        annotation: Annotation = current_file["annotation"]
        num_speakers: int = len(annotation.labels())

        if self.num_speakers is not None and self.num_speakers != num_speakers:
            return False

        if self.min_speakers is not None and self.min_speakers > num_speakers:
            return False

        if self.max_speakers is not None and self.max_speakers < num_speakers:
            return False

        return True
