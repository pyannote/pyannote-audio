#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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
# Pavel KORSHUNOV - https://www.idiap.ch/~pkorshunov/
# Paul LERNER
# Vincent BRIGNATZ
# Alexis PLAQUET

"""Custom protocols

Protocols:
  MyDatabase:
    Collection:
      MyProtocol:
        train:
          uris: xxx.lst
          annotation: xxx.rttm
          annotated: xxx.uem
"""

from pathlib import Path
import string


from . import protocol as protocol_module

from pyannote.database.protocol.protocol import ProtocolFile


import warnings
from numbers import Number
from typing import Text, Dict, Callable, Any, Union
import functools

from .protocol.protocol import Subset
from .protocol.segmentation import SegmentationProtocol
from .protocol.speaker_diarization import SpeakerDiarizationProtocol

from importlib.metadata import entry_points

from .util import get_annotated

from .loader import load_lst, load_trial

# All "Loader" classes types (eg RTTMLoader, UEMLoader, ...) retrieved from the entry point.
LOADERS = {
    ep.name: ep
    for ep in entry_points(group="pyannote.database.loader")
}


def Template(template: Text, database_yml: Path) -> Callable[[ProtocolFile], Any]:
    """Get data loader based on template

    Parameters
    ----------
    template : str
        Path format template (e.g. "/path/to/{uri}.csv").
        Extension (here ".csv") determined which data loader to use.
    database_yml : Path
        Path to YAML configuration file, to which `template` is relative.
        Defaults to assume that `template` is absolute or relative to 
        current working directory.

    Returns
    -------
    data_loader : Callable[[ProtocolFile], Any]
        Callable that takes a ProtocolFile and returns some data.

    See also
    --------
    pyannote.database.loader
    """

    path = Path(template)
    if path.suffix not in LOADERS:
        msg = f"No loader for files with '{path.suffix}' suffix"
        raise ValueError(msg)

    Loader = LOADERS[path.suffix].load()

    def load(current_file: ProtocolFile):
        path = resolve_path(Path(template.format(**abs(current_file))), database_yml)

        # check if file exists
        if not path.is_file():
            msg = f"No such file or directory: '{path}' (via '{template}' template)."
            raise FileNotFoundError(msg)

        loader = Loader(path)
        return loader(current_file)

    return load


def NumericValue(value):
    def load(current_file: ProtocolFile):
        return value
    return load


def resolve_path(path: Path, database_yml: Path) -> Path:
    """Resolve path

    Parameters
    ----------
    path : `Path`
        Path. Can be either absolute, relative to current working directory, 
        or relative to `database_yml` parent directory.
    database_yml : `Path`
        Path to YAML configuration file. 

    Returns
    -------
    resolved_path: `Path`
        Resolved path.
    """

    path = path.expanduser()

    if path.is_file():
        return path

    else:
        relative_path = database_yml.parent / path
        if relative_path.is_file():
            return relative_path

    msg = f'Could not find file "{path}".'
    raise FileNotFoundError(msg)


def meta_subset_iter(
    meta_database: Text,
    meta_task: Text,
    meta_protocol: Text,
    meta_subset: Subset,
    subset_entries: Dict,
    database_yml: Path,
):
    """Meta-protocol method that iterates over a subset

    Parameters
    ----------
    meta_database : str
        "X"
    meta_task : str
        Task name (e.g. SpeakerDiarization, SpeakerVerification)
    meta_protocol : str
        Protocol name (e.g. MyProtocol)
    meta_subset : {"train", "development", "test"}
        Subset
    subset_entries : dict
        Subset entries.
            Etape.SpeakerDiarization.TV: [train]
            REPERE.SpeakerDiarization.Phase1: [train, development]
            REPERE.SpeakerDiarization.Phase2: [train, development]
    """

    # this is imported here to avoid circular imports
    from . import registry

    for protocol, subsets in subset_entries.items():
        partial_protocol = registry.get_protocol(protocol)
        for subset in subsets:
            method_name = f"{subset}_iter"
            for file in getattr(partial_protocol, method_name)():
                yield file


def gather_loaders(
    entries: Dict,
    database_yml: Path,
) -> dict:
    """Loads all Loaders for data type specified in 'entries' into a dict.

    Parameters
    ----------
    entries : Dict, optional
        Subset entries (eg 'uri', 'annotated', 'annotation', ...)
    database_yml : Path, optional
        Path to the 'database.yml' file

    Returns
    -------
    dict
        A dictionary mapping each key of entry (except 'uri' and 'trial')
        to a function that given a ProtocolFile returns the data type
        related to this entry.
    """
    lazy_loader = dict()

    for key, value in entries.items():

        if key == "uri" or key == "trial":
            continue

        if isinstance(value, Number):
            lazy_loader[key] = NumericValue(value)
            continue

        # check whether value (path) contains placeholders such as {uri} or {subset}
        _, placeholders, _, _ = zip(*string.Formatter().parse(value))
        is_template = len(set(placeholders) - set([None])) > 0

        if is_template:

            # make sure old database.yml specifications still work but warn the user
            # that they can now get rid of this "_" prefix
            if value.startswith("_"):
                msg = (
                    "Since version 4.1, pyannote.database is smart enough to know "
                    "when paths defined in 'database.yml' contains placeholders. "
                    "Remove the underscore (_) prefix to get rid of this warning."
                )
                warnings.warn(msg)
                value = value[1:]

            lazy_loader[key] = Template(value, database_yml)

        else:

            path = resolve_path(Path(value), database_yml)

            # check if file exists
            if not path.is_file():
                msg = f"No such file or directory: '{path}'"
                raise FileNotFoundError(msg)

            # check if loader exists
            if path.suffix not in LOADERS:
                msg = f"No loader for file with '{path.suffix}' suffix"
                raise TypeError(msg)

            # load custom loader class
            Loader = LOADERS[path.suffix].load()

            # TODO: As it is right now, every call to "subset_iter" also calls "Loader(path)".
            # However, calling "Loader(path)" might be time consuming so we should probably cache it:
            # Current behavior:
            #   for _ in protocol.train(): pass   # first call is slow (compute Loader(path))
            #   for _ in protocol.train(): pass   # subsequent calls are equally slow (compute Loader(path))
            # Proposed behavior:
            #   for _ in protocol.train(): pass   # first call is slow (compute and cache Loader(path))
            #   for _ in protocol.train(): pass   # subsequent calls are fast (use cached Loader(path))
            lazy_loader[key] = Loader(path)
    return lazy_loader


def subset_iter(
    self,
    database: Text = None,
    task: Text = None,
    protocol: Text = None,
    subset: Subset = None,
    entries: Dict = None,
    database_yml: Path = None,
    **metadata,
):
    """

    Parameters
    ----------
    database : str
        Database name (e.g. MyDatabase)
    task : str
        Task name (e.g. SpeakerDiarization, SpeakerVerification)
    protocol : str
        Protocol name (e.g. MyProtocol)
    subset : {"train", "development", "test"}
        Subset
    entries : dict
        Subset entries.
    database_yml : `Path`
        Path to the 'database.yml' file
    metadata : dict
        Additional metadata to be added to each ProtocolFile (such
        as "scope" or "classes")
    """

    if "uri" in entries:
        uri = entries["uri"]

    elif "uris" in entries:
        uri = entries["uris"]
        msg = (
            f"Found deprecated 'uris' entry in {database}.{task}.{protocol}.{subset}. "
            f"Please use 'uri' (singular) instead, in '{database_yml}'."
        )
        warnings.warn(msg, DeprecationWarning)

    else:
        msg = f"Missing mandatory 'uri' entry in {database}.{task}.{protocol}.{subset}"
        raise ValueError(msg)

    uris = load_lst(resolve_path(Path(uri), database_yml))

    lazy_loader = gather_loaders(entries=entries, database_yml=database_yml)

    for uri in uris:
        yield ProtocolFile(
            {"uri": uri, "database": database, "subset": subset, **metadata}, lazy=lazy_loader
        )

def subset_trial(
    self,
    database: Text = None,
    task: Text = None,
    protocol: Text = None,
    subset: Subset = None,
    entries: Dict = None,
    database_yml: Path = None,
):
    """

    Parameters
    ----------
    database : str
        Database name (e.g. MyDatabase)
    task : str
        Task name (e.g. SpeakerDiarization, SpeakerVerification)
    protocol : str
        Protocol name (e.g. MyProtocol)
    subset : {"train", "development", "test"}
        Subset
    entries : dict
        Subset entries.
    database_yml : `Path`
        Path to the 'database.yml' file
    """

    lazy_loader = gather_loaders(entries=entries, database_yml=database_yml)
    lazy_loader["try_with"] = get_annotated

    # meant to store and cache one `ProtocolFile` instance per file
    files: Dict[Text, ProtocolFile] = dict()

    # iterate trials and use preloaded test files
    for trial in load_trial(resolve_path(Path(entries["trial"]), database_yml)):
        # create `ProtocolFile` only the first time this uri is encountered
        uri1, uri2 = trial["uri1"], trial["uri2"]
        if uri1 not in files:
            files[uri1] = self.preprocess(
                ProtocolFile(
                    {"uri": uri1, "database": database, "subset": subset},
                    lazy=lazy_loader,
                )
            )
        if uri2 not in files:
            files[uri2] = self.preprocess(
                ProtocolFile(
                    {"uri": uri2, "database": database, "subset": subset},
                    lazy=lazy_loader,
                )
            )

        yield {
            "reference": trial["reference"],
            "file1": files[uri1],
            "file2": files[uri2],
        }


def get_init(protocols):
    def init(self):
        super(self.__class__, self).__init__()
        for protocol in protocols:
            self.register_protocol(*protocol)

    return init


def get_custom_protocol_class_name(database: Text, task: Text, protocol: Text):
    return f"{database}__{task}__{protocol}"


def create_protocol(
    database: Text,
    task: Text,
    protocol: Text,
    protocol_entries: Dict,
    database_yml: Path,
    version: dict | None = None,
) -> Union[type, None]:
    """Create new protocol class

    Parameters
    ----------
    database : str
    task : str
    protocol : str
    protocol_entries : dict
    version : dict, optional
        Annotation version as {"name": ..., "metadata": ...}

    Returns
    -------
    CustomProtocol : type or None

    """

    
    try:
        base_class = getattr(
            protocol_module, "Protocol" if task == "Protocol" else f"{task}Protocol"
        )

    except AttributeError:
        msg = (
            f"Ignoring '{database}.{task}' protocols found in {database_yml} "
            f"because '{task}' tasks are not supported yet."
        )
        print(msg)
        return None

    # Collections do not define subsets, so we artificially create one (called "files")
    #
    #    MyCollection:
    #      uri: /path/to/collection.lst
    #
    # becomes
    #
    #    MyCollection:
    #      files:
    #        uri: /path/to/collection.lst
    if task == "Collection":
        protocol_entries = {"files": protocol_entries}

    metadata = dict()

    if version is not None:
        metadata["version"] = version

    if issubclass(base_class, SegmentationProtocol):
        if "classes" in protocol_entries:
            metadata["classes"] = protocol_entries.pop("classes")

    if issubclass(base_class, SpeakerDiarizationProtocol):
        scope = protocol_entries.pop("scope", None)

        if scope is None and database != "X":
            msg = (
                f"'{database}.{task}.{protocol}' found in {database_yml} does not define "
                f"the 'scope' of speaker labels (file, database, or global). Setting it to 'file'."
            )   
            print(msg)
            metadata["scope"] = "file"

        else:
            metadata["scope"] = scope

    methods = dict()
    for subset, subset_entries in protocol_entries.items():

        if subset not in [
            "files",
            "train",
            "development",
            "test",
            "train_trial",
            "development_trial",
            "test_trial",
        ]:
            msg = (
                f"Ignoring '{database}.{task}.{protocol}.{subset}' found in {database_yml} "
                f"because '{subset}' entries are not supported yet."
            )
            warnings.warn(msg)
            continue

        # "version" is reserved for the annotation version (see `metadata`)
        if isinstance(subset_entries, dict) and "version" in subset_entries:
            msg = (
                f"'{database}.{task}.{protocol}.{subset}' ({database_yml}) defines "
                f"its own 'version' entry, but 'version' is reserved for annotation "
                f"versioning."
            )
            raise ValueError(msg)

        method_name = f"{subset}_iter"
        if database == "X":
            methods[method_name] = staticmethod(
                functools.partial(
                    meta_subset_iter,
                    database,
                    task,
                    protocol,
                    subset,
                    subset_entries,
                    database_yml,
                )
            )
        else:

            methods[method_name] = functools.partialmethod(
                subset_iter,
                database=database,
                task=task,
                protocol=protocol,
                subset=subset,
                entries=subset_entries,
                database_yml=database_yml,
                **metadata,
            )

            if "trial" in subset_entries.keys():
                methods[f"{subset}_trial"] = functools.partialmethod(
                    subset_trial,
                    database=database,
                    task=task,
                    protocol=protocol,
                    subset=subset,
                    entries=subset_entries,
                    database_yml=database_yml,
                )

    #  making custom protocol pickable by adding it to pyannote.database.custom module
    custom_protocol_class_name = get_custom_protocol_class_name(
        database, task, protocol
    )
    CustomProtocolClass = type(custom_protocol_class_name, (base_class,), methods)
    globals()[custom_protocol_class_name] = CustomProtocolClass

    return CustomProtocolClass
