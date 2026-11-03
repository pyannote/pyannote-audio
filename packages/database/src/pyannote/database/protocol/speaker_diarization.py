# The MIT License (MIT)

# Copyright (c) 2016- CNRS

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

from .segmentation import SegmentationProtocol


class SpeakerDiarizationProtocol(SegmentationProtocol):
    """A protocol for speaker diarization experiments

    A speaker diarization protocol can be defined programmatically by creating
    a class that inherits from SpeakerDiarizationProtocol and implements at
    least one of `train_iter`, `development_iter` and `test_iter` methods:

        >>> class MySpeakerDiarizationProtocol(SpeakerDiarizationProtocol):
        ...     def train_iter(self) -> Iterator[Dict]:
        ...         yield {"uri": "filename1",
        ...                "annotation": Annotation(...),
        ...                "annotated": Timeline(...)}
        ...         yield {"uri": "filename2",
        ...                "annotation": Annotation(...),
        ...                "annotated": Timeline(...)}

    `{subset}_iter` should return an iterator of dictionnaries with
        - "uri" key (mandatory) that provides a unique file identifier (usually
          the filename),
        - "annotation" key (mandatory for train and development subsets) that
          provides reference speaker diarization as a `pyannote.core.Annotation`
          instance,
        - "annotated" key (recommended) that describes which part of the file
          has been annotated, as a `pyannote.core.Timeline` instance. Any part
          of "annotation" that lives outside of the provided "annotated" will
          be removed. This is also used by `pyannote.metrics` to remove
          un-annotated regions from its evaluation report, and by
          `pyannote.audio` to not consider empty un-annotated regions as
          non-speech.
        - any other key that the protocol may provide.

    It can then be used in Python like this:

        >>> protocol = MySpeakerDiarizationProtocol()
        >>> for file in protocol.train():
        ...    print(file["uri"])
        filename1
        filename2

    A speaker diarization protocol can also be defined using `pyannote.database`
    configuration file, whose (configurable) path defaults to "~/database.yml".

    ~~~ Content of ~/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Protocols:
      MyDatabase:
        SpeakerDiarization:
          MyProtocol:
            train:
                uri: /path/to/collection.lst
                annotation: /path/to/reference.rttm
                annotated: /path/to/reference.uem
                any_other_key: ... # see custom loader documentation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    where "/path/to/collection.lst" contains the list of identifiers of the
    files in the collection:

    ~~~ Content of "/path/to/collection.lst ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1
    filename2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    "/path/to/reference.rttm" contains the reference speaker diarization using
    RTTM format:

    ~~~ Content of "/path/to/reference.rttm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SPEAKER filename1 1 3.168 0.800 <NA> <NA> speaker_A <NA> <NA>
    SPEAKER filename1 1 5.463 0.640 <NA> <NA> speaker_A <NA> <NA>
    SPEAKER filename1 1 5.496 0.574 <NA> <NA> speaker_B <NA> <NA>
    SPEAKER filename1 1 10.454 0.499 <NA> <NA> speaker_B <NA> <NA>
    SPEAKER filename2 1 2.977 0.391 <NA> <NA> speaker_C <NA> <NA>
    SPEAKER filename2 1 18.705 0.964 <NA> <NA> speaker_C <NA> <NA>
    SPEAKER filename2 1 22.269 0.457 <NA> <NA> speaker_A <NA> <NA>
    SPEAKER filename2 1 28.474 1.526 <NA> <NA> speaker_A <NA> <NA>
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    "/path/to/reference.uem" describes the annotated regions using UEM format:

    ~~~ Content of "/path/to/reference.uem ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filename1 NA 0.000 30.000
    filename2 NA 0.000 30.000
    filename2 NA 40.000 70.000
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    It can then be used in Python like this:

        >>> from pyannote.database import registry
        >>> protocol = registry.get_protocol('MyDatabase.SpeakerDiarization.MyProtocol')
        >>> for file in protocol.train():
        ...    print(file["uri"])
        filename1
        filename2
    """

    pass
