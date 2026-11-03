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


import functools

from pyannote.core import Annotation, Segment, Timeline

from .protocol import Preprocessor, Preprocessors, Protocol, ProtocolFile, Subset


def crop_annotated(
    current_file: ProtocolFile, existing_preprocessor: Preprocessor | None = None
) -> Timeline:
    """Preprocessor that crops 'annotated' according to 'duration'

    Returns 'annotated' unchanged if 'duration' is not available

    Parameters
    ----------
    current_file : ProtocolFile
        Protocol file.
    existing_preprocessor : Preprocessor, optional
        When provided, this preprocessor must be used to get the initial
        'annotated' instead of getting it from 'current_file["annotated"]'

    Returns
    -------
    cropped_annotated : Timeline
        "annotated" cropped by "duration".
    """

    if existing_preprocessor is None:
        annotated = current_file.get("annotated", None)
    else:
        annotated = existing_preprocessor(current_file)

    if annotated is None:
        return None

    duration = current_file.get("duration", None)
    if duration is None:
        return annotated

    # crop 'annotated' to 'duration'
    duration = Segment(0.0, duration)

    if annotated and annotated.extent() not in duration:
        return annotated.crop(duration, mode="intersection")

    return annotated


def crop_annotation(
    current_file: ProtocolFile, existing_preprocessor: Preprocessor | None = None
) -> Annotation:
    """Preprocessor that crops 'annotation' by 'annotated'

    Returns 'annotation' unchanged if 'annotated' is not available

    Parameters
    ----------
    current_file : ProtocolFile
        Protocol file.
    existing_preprocessor : Preprocessor, optional
        When provided, this preprocessor must be used to get the initial
        'annotation' instead of getting it from 'current_file["annotation"]'

    Returns
    -------
    cropped_annotation : Annotation
        "annotation" cropped by "annotated".
    """

    if existing_preprocessor is None:
        annotation = current_file.get("annotation", None)
    else:
        annotation = existing_preprocessor(current_file)

    if annotation is None:
        return None

    annotated = current_file.get("annotated", None)
    if annotated is None:
        return annotation

    # crop 'annotation' to 'annotated' extent
    if annotated and not annotated.covers(annotation.get_timeline()):
        return annotation.crop(annotated, mode="intersection")

    return annotation


class SegmentationProtocol(Protocol):
    """A protocol for segmentation experiments

    A segmentation protocol can be defined programmatically by creating
    a class that inherits from SegmentationProtocol and implements at
    least one of `train_iter`, `development_iter` and `test_iter` methods:

        >>> class MySegmentationProtocol(SegmentationProtocol):
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
          provides reference segmentation as a `pyannote.core.Annotation`
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

        >>> protocol = MySegmentationProtocol()
        >>> for file in protocol.train():
        ...    print(file["uri"])
        filename1
        filename2

    A segmentation protocol can also be defined using `pyannote.database`
    configuration file, whose (configurable) path defaults to "~/database.yml".

    ~~~ Content of ~/database.yml ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Protocols:
      MyDatabase:
        Segmentation:
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

    "/path/to/reference.rttm" contains the reference segmentation using
    RTTM format:

    ~~~ Content of "/path/to/reference.rttm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SPEAKER filename1 1 3.168 0.800 <NA> <NA> music <NA> <NA>
    SPEAKER filename1 1 5.463 0.640 <NA> <NA> music <NA> <NA>
    SPEAKER filename1 1 5.496 0.574 <NA> <NA> speech <NA> <NA>
    SPEAKER filename1 1 10.454 0.499 <NA> <NA> speech <NA> <NA>
    SPEAKER filename2 1 2.977 0.391 <NA> <NA> noise <NA> <NA>
    SPEAKER filename2 1 18.705 0.964 <NA> <NA> noise <NA> <NA>
    SPEAKER filename2 1 22.269 0.457 <NA> <NA> music <NA> <NA>
    SPEAKER filename2 1 28.474 1.526 <NA> <NA> music <NA> <NA>
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

    def __init__(self, preprocessors: Preprocessors | None = None):

        if preprocessors is None:
            preprocessors = dict()

        # wrap exisiting "annotated" preprocessor by crop_annotated so that
        # "annotated" is automatically cropped by file "duration" when provided
        preprocessors["annotated"] = functools.partial(
            crop_annotated, existing_preprocessor=preprocessors.get("annotated", None)
        )

        # wrap exisiting "annotation" preprocessor by crop_annotation so that
        # "annotation" is automatically cropped by "annotated" when provided
        preprocessors["annotation"] = functools.partial(
            crop_annotation, existing_preprocessor=preprocessors.get("annotation", None)
        )

        super().__init__(preprocessors=preprocessors)

    def stats(self, subset: Subset = "train") -> dict:
        """Obtain global statistics on a given subset

        Parameters
        ----------
        subset : {'train', 'development', 'test'}

        Returns
        -------
        stats : dict
            Dictionary with the followings keys:
            * annotated: float
            total duration (in seconds) of the parts that were manually annotated
            * annotation: float
            total duration (in seconds) of actual annotations
            * n_files: int
            number of files in the subset
            * labels: dict
            maps classes with their total duration (in seconds)
        """

        from ..util import get_annotated

        annotated_duration = 0.0
        annotation_duration = 0.0
        n_files = 0
        labels = {}

        for item in getattr(self, subset)():
            annotated = get_annotated(item)
            annotated_duration += annotated.duration()

            # increment 'annotation' total duration
            annotation = item["annotation"]
            annotation_duration += annotation.get_timeline().duration()

            for label, duration in annotation.chart():
                if label not in labels:
                    labels[label] = 0.0
                labels[label] += duration
            n_files += 1

        stats = {
            "annotated": annotated_duration,
            "annotation": annotation_duration,
            "n_files": n_files,
            "labels": labels,
        }

        return stats
