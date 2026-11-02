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
# Alexis PLAQUET

"""pyannote.database"""

import importlib.metadata
import warnings

from .database import Database
from .file_finder import FileFinder
from .protocol.protocol import Preprocessors, Protocol, ProtocolFile, Subset
from .registry import LoadingMode, registry
from .util import get_annotated, get_label_identifier, get_unique_identifier

__version__ = importlib.metadata.version("pyannote-database")


def get_protocol(name, preprocessors: Preprocessors | None = None) -> Protocol:
    """Get protocol by full name

    name : str
        Protocol full name (e.g. "Etape.SpeakerDiarization.TV")
    preprocessors : dict or (key, preprocessor) iterable
        When provided, each protocol item (dictionary) are preprocessed, such
        that item[key] = preprocessor(item). In case 'preprocessor' is not
        callable, it should be a string containing placeholder for item keys
        (e.g. {'audio': '/path/to/{uri}.wav'})

    Returns
    -------
    protocol : Protocol
        Protocol instance
    """
    warnings.warn(
        "`get_protocol` has been deprecated in favor of `pyannote.database.registry.get_protocol`.",
        DeprecationWarning,
    )
    return registry.get_protocol(name, preprocessors=preprocessors)


__all__ = [
    "registry",
    "get_protocol",
    "LoadingMode",
    "Database",
    "Protocol",
    "ProtocolFile",
    "Subset",
    "FileFinder",
    "get_annotated",
    "get_unique_identifier",
    "get_label_identifier",
]
