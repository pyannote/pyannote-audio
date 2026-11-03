# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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


from collections.abc import Iterator

from .protocol import LEGACY_SUBSET_MAPPING, Subset
from .speaker_diarization import SpeakerDiarizationProtocol


class SpeakerVerificationProtocol(SpeakerDiarizationProtocol):
    """A protocol for speaker verification experiments

    A speaker verification protocol can be defined programmatically by creating
    a class that inherits from `SpeakerVerificationProtocol` and implement at
    least one of `train_trial_iter`, `development_trial_iter` and
    `test_trial_iter` methods:

        >>> class MySpeakerVerificationProtocol(SpeakerVerificationProtocol):
        ...     def train_trial_iter(self) -> Iterator[Dict]:
        ...         yield {"reference": 0,
        ...                "file1": {
        ...                     "uri":"filename1",
        ...                     "try_with":Timeline(...)
        ...                },
        ...                "file2": {
        ...                     "uri":"filename3",
        ...                     "try_with":Timeline(...)
        ...                },
        ...         }

    `{subset}_trial_iter` should return an iterator of dictionnaries with

    - `reference` key (mandatory) that provides an int portraying whether
      `file1` and `file2` are uttered by the same speaker (1 is same, 0 is
      different),
    - `file1` key (mandatory) that provides the first file,
    - `file2` key (mandatory) that provides the second file.

    Both `file1` and `file2` should be provided as dictionaries or ProtocolFile
    instances with

    - `uri` key (mandatory),
    - `try_with` key (mandatory) that describes which part of the file should
      be used in the validation process, as a `pyannote.core.Timeline` instance.
    - any other key that the protocol may provide.

    It can then be used in Python like this:

        >>> protocol = MySpeakerVerificationProtocol()
        ... for trial in protocol.train_trial():
        ...     print(f"{trial['reference']} {trial['file1']['uri']} {trial['file2']['uri']}")
        1 filename1 filename2
        0 filename1 filename3

    A speaker verification protocol can also be defined using `pyannote.database`
    configuration file, whose (configurable) path defaults to "~/database.yml".

    ~~~ Content of ~/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Protocols:
        MyDatabase:
        SpeakerVerification:
            MyProtocol:
            train:
                uri: /path/to/train.lst
                duration: /path/to/duration.map
                trial: /path/to/trial.txt
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    where `/path/to/train.lst` contains the list of identifiers of the
    files in the collection:

    ~~~ Content of /path/to/train.lst~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1
    filename2
    filename3
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    `/path/to/duration.map` contains the duration of the files:

    ~~~ Content of /path/to/duration.map ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1 30.000
    filename2 30.000
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    `/path/to/trial.txt` contains a list of trials :

    ~~~ Content of /path/to/trial ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 filename1 filename2
    0 filename1 filename3
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    `1` stands for _target_ trials and `0` for _non-target_ trials.
    In the example below, it means that the same speaker uttered files
    `filename1` and `filename2` and that `filename1` and `filename3` are from
    two different speakers.

    It can then be used in Python like this:

        >>> from pyannote.database import registry
        >>> protocol = registry.get_protocol('MyDatabase.SpeakerVerification.MyProtocol')
        >>> for trial in protocol.train_trial():
        ...     print(f"{trial['reference']} {trial['file1']['uri']} {trial['file2']['uri']}")
        1 filename1 filename2
        0 filename1 filename3

    Note that speaker verification protocols (`SpeakerVerificationProtocol`)
    are a subclass of speaker diarization protocols (`SpeakerDiarizationProtocol`).
    As such, they also define regular `{subset}` methods.
    """

    def subset_trial_helper(self, subset: Subset) -> Iterator[dict]:

        try:
            trials = getattr(self, f"{subset}_trial_iter")()
        except (AttributeError, NotImplementedError):
            # previous pyannote.database versions used `trn_try_iter` instead
            # of `train_trial_iter`, `dev_try_iter` instead of
            # `development_trial_iter`, and `tst_try_iter` instead of
            # `test_iter`. therefore, we use the legacy version when it is
            # available (and the new one is not).
            subset_legacy = LEGACY_SUBSET_MAPPING[subset]
            try:
                trials = getattr(self, f"{subset_legacy}_try_iter")()
            except AttributeError:
                msg = f"{subset}_trial_iter is not implemented."
                raise AttributeError(msg)

        for trial in trials:
            trial["file1"] = self.preprocess(trial["file1"])
            trial["file2"] = self.preprocess(trial["file2"])
            yield trial

    def train_trial_iter(self) -> Iterator[dict]:
        """Iterate over trials in the train subset"""
        raise NotImplementedError()

    def development_trial_iter(self) -> Iterator[dict]:
        """Iterate over trials in the development subset"""
        raise NotImplementedError()

    def test_trial_iter(self) -> Iterator[dict]:
        """Iterate over trials in the test subset"""
        raise NotImplementedError()

    def train_trial(self) -> Iterator[dict]:
        return self.subset_trial_helper("train")

    def development_trial(self) -> Iterator[dict]:
        return self.subset_trial_helper("development")

    def test_trial(self) -> Iterator[dict]:
        return self.subset_trial_helper("test")
