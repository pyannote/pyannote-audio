# The MIT License (MIT)

# Copyright (c) 2020-2025 CNRS
# Copyright (c) 2025- pyannoteAI

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
# Hervé BREDIN
# Vincent BRIGNATZ

"""Data loaders"""

import string
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from pyannote.core import Annotation, Timeline

from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.database.util import load_lab, load_rttm, load_uem

try:
    import meeteval.io
    from meeteval.io.seglst import SegLST

    MEETEVAL_IS_AVAILABLE = True

except ImportError:
    MEETEVAL_IS_AVAILABLE = False


def load_lst(file_lst):
    """Load LST file

    LST files provide a list of URIs (one line per URI)

    Parameter
    ---------
    file_lst : `str`
        Path to LST file.

    Returns
    -------
    uris : `list`
        List or uris
    """

    with open(file_lst) as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]


def load_trial(file_trial):
    """Load trial file

    Trial files provide a list of two URIs and their reference

    Parameter
    ---------
    file_trial : `str`
        Path to trial file.

    Returns
    -------
    list_trial : `list`
        List of trial
    """

    trials = pd.read_table(file_trial, sep=r"\s+", names=["reference", "uri1", "uri2"])

    for _, reference, uri1, uri2 in trials.itertuples():
        yield {"reference": reference, "uri1": uri1, "uri2": uri2}


class RTTMLoader:
    """RTTM loader

    Can be used as a preprocessor.

    Parameters
    ----------
    path : str
        Path to RTTM file with optional ProtocolFile key placeholders
        (e.g. "/path/to/{database}/{subset}/{uri}.rttm")
    """

    def __init__(self, path: str = None):
        super().__init__()

        self.path = str(path)

        _, placeholders, _, _ = zip(*string.Formatter().parse(self.path))
        self.placeholders_ = set(placeholders) - set([None])
        self.loaded_ = dict() if self.placeholders_ else load_rttm(self.path)

    def __call__(self, file: ProtocolFile) -> Annotation:
        uri = file["uri"]

        if uri in self.loaded_:
            return self.loaded_[uri]

        sub_file = {key: file[key] for key in self.placeholders_}
        loaded = load_rttm(self.path.format(**sub_file))
        if uri not in loaded:
            loaded[uri] = Annotation(uri=uri)

        # do not cache annotations when there is one RTTM file per "uri"
        # since loading it should be quite fast
        if "uri" in self.placeholders_:
            return loaded[uri]

        # when there is more than one file in loaded RTTM, cache them all
        # so that loading future "uri" will be instantaneous
        self.loaded_.update(loaded)

        return self.loaded_[uri]


class STMLoader:
    """STM loader

    Parameters
    ----------
    path : str
        Path to STM file with optional AudioFile key placeholders
        (e.g. "/path/to/{database}/{subset}/{uri}.stm")
    """

    def __init__(self, path: str | Path | None = None):
        super().__init__()

        self.path = str(path)

        _, placeholders, _, _ = zip(*string.Formatter().parse(self.path))
        self.placeholders_ = set(placeholders) - set([None])

        if self.placeholders_:
            self.loaded_: dict[str, SegLST] = dict()
            return

        if MEETEVAL_IS_AVAILABLE:
            seglst: SegLST = meeteval.io.load(self.path, format="stm").to_seglst()
            session_ids = set(s["session_id"] for s in seglst)
            self.loaded_: dict[str, SegLST] = {
                session_id: SegLST([s for s in seglst if s["session_id"] == session_id])
                for session_id in session_ids
            }

            return

        warnings.warn("MeetEval is not available, STM files cannot be loaded.")
        self.loaded_: dict[str, SegLST] = dict()

    def __call__(self, file: ProtocolFile) -> "SegLST":
        uri = file["uri"]

        if uri in self.loaded_:
            return self.loaded_[uri]

        sub_file = {key: file[key] for key in self.placeholders_}

        if MEETEVAL_IS_AVAILABLE:
            seglst: SegLST = meeteval.io.load(
                self.path.format(**sub_file), format="stm"
            ).to_seglst()
            session_ids = set(s["session_id"] for s in seglst)
            loaded: dict[str, SegLST] = {
                session_id: SegLST([s for s in seglst if s["session_id"] == session_id])
                for session_id in session_ids
            }
        else:
            warnings.warn("MeetEval is not available, STM files cannot be loaded.")
            loaded = dict()

        if uri not in loaded:
            if MEETEVAL_IS_AVAILABLE:
                loaded[uri] = SegLST([])
            else:
                loaded[uri] = None

        # do not cache transcription when there is one STM file per "uri"
        # since loading it should be quite fast
        if "uri" in self.placeholders_:
            return loaded[uri]

        # when there is more than one file in loaded STM, cache them all
        # so that loading future "uri" will be instantaneous
        self.loaded_.update(loaded)

        return self.loaded_[uri]


class UEMLoader:
    """UEM loader

    Can be used as a preprocessor.

    Parameters
    ----------
    path : str
        Path to UEM file with optional ProtocolFile key placeholders
        (e.g. "/path/to/{database}/{subset}/{uri}.uem")
    """

    def __init__(self, path: str = None):
        super().__init__()

        self.path = str(path)

        _, placeholders, _, _ = zip(*string.Formatter().parse(self.path))
        self.placeholders_ = set(placeholders) - set([None])
        self.loaded_ = dict() if self.placeholders_ else load_uem(self.path)

    def __call__(self, file: ProtocolFile) -> Timeline:
        uri = file["uri"]

        if uri in self.loaded_:
            return self.loaded_[uri]

        sub_file = {key: file[key] for key in self.placeholders_}
        loaded = load_uem(self.path.format(**sub_file))
        if uri not in loaded:
            loaded[uri] = Timeline(uri=uri)

        # do not cache timelines when there is one UEM file per "uri"
        # since loading it should be quite fast
        if "uri" in self.placeholders_:
            return loaded[uri]

        # when there is more than one file in loaded UEM, cache them all
        # so that loading future "uri" will be instantaneous
        self.loaded_.update(loaded)

        return self.loaded_[uri]


class LABLoader:
    """LAB loader

    Parameters
    ----------
    path : str
        Path to LAB file with mandatory {uri} placeholder.
        (e.g. "/path/to/{uri}.lab")

        each .lab file contains the segments for a single audio file, in the following format:
        start end label

        ex.
        0.0 12.3456 sing
        12.3456 15.0 nosing
        ...
    """

    def __init__(self, path: str = None):
        super().__init__()

        self.path = str(path)

        _, placeholders, _, _ = zip(*string.Formatter().parse(self.path))
        self.placeholders_ = set(placeholders) - set([None])
        if "uri" not in self.placeholders_:
            raise ValueError("`path` must contain the {uri} placeholder.")

    def __call__(self, file: ProtocolFile) -> Annotation:
        uri = file["uri"]

        sub_file = {key: file[key] for key in self.placeholders_}
        return load_lab(self.path.format(**sub_file), uri=uri)


class MAPLoader:
    """Mapping loader

    For generic files with format :
    {uri} {value}

    Exemples :

        duration.map :

            filename1 60.0
            filename2 123.450
            filename3 32.400

        domain.map :

            filename1 radio
            filename2 radio
            filename3 phone

    Parameter
    ---------
    map : Path
        Path to mapping file
    """

    def __init__(self, mapping: Path):
        self.mapping = mapping

        names = ["uri", "value"]
        dtype = {
            "uri": str,
        }
        self.data_ = pd.read_csv(mapping, names=names, dtype=dtype, sep=r"\s+")

        # get colum 'value' dtype, allowing us to acces it during subset
        self.dtype = self.data_.dtypes["value"]

        if self.data_.duplicated(["uri"]).any():
            print(f"Found following duplicate key in file {mapping}")
            print(self.data_[self.data_.duplicated(["uri"], keep=False)])
            raise ValueError()

        self.data_ = self.data_.groupby("uri")

    def __call__(self, current_file: ProtocolFile) -> Any:
        uri = current_file["uri"]

        try:
            value = self.data_.get_group(uri).value.item()
        except KeyError:
            msg = f"Couldn't find mapping for {uri} in {self.mapping}"
            raise KeyError(msg)

        return value
