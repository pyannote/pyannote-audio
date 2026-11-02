#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2022- CNRS
# Copyright (c) 2022- Université Paul Sabatier

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

from enum import Enum
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Text, Tuple, Type, Union
import warnings

from pyannote.database.protocol.protocol import Preprocessors, Protocol
from .custom import create_protocol, get_init
from .database import Database
import yaml


# controls what to do in case of protocol name conflict
class LoadingMode(Enum):
    OVERRIDE = 0  # override existing protocol
    KEEP = 1  # keep existing protocol
    ERROR = 2  # raise an error


def _apply_version(entries: Dict, version: Text) -> Dict:
    """Replace {version} placeholder in protocol entries

    Parameters
    ----------
    entries : dict
        Protocol entries (e.g. {"scope": "file", "train": {"uri": ...}}).
    version : str
        Version name (e.g. "original").

    Returns
    -------
    versioned_entries : dict
        Copy of `entries` where {version} is replaced in subset keys and
        string values. Other entries (e.g. "scope") are left unchanged.
    """

    def sub(value):
        return value.replace("{version}", version) if isinstance(value, str) else value

    return {
        key: {sub(k): sub(v) for k, v in value.items()} if isinstance(value, dict) else value
        for key, value in entries.items()
    }


def _expand_versions(
    db_name: Text,
    task_name: Text,
    protocol: Text,
    protocol_entries: Dict,
    database_yml: Path,
) -> list[tuple[str, dict, dict | None]]:
    """Expand protocol into one protocol per version

    Parameters
    ----------
    db_name : str
        Database name (e.g. "MyDatabase").
    task_name : str
        Task name (e.g. "SpeakerDiarization").
    protocol : str
        Protocol name (e.g. "MyProtocol").
    protocol_entries : dict
        Protocol entries, as found in `database_yml`.
    database_yml : Path
        Path to YAML configuration file (used in error messages).

    Returns
    -------
    versioned_protocols : list of ("<protocol>@<name>", entries, version) tuples, one
    per declared version, the default (first listed) version first, where
    version is {"name": name, "metadata": metadata}.
    Protocol blocks without a `versions:` key are returned as
    [(protocol, entries, None)], i.e. the exact unversioned behavior.

    Raises
    ------
    ValueError
        If `versions:` does not follow the expected indentation format,
        or if a version name contains ".".
    """

    where = f"'{db_name}.{task_name}.{protocol}' ({database_yml})"

    if not isinstance(protocol_entries, dict) or "versions" not in protocol_entries:
        return [(protocol, protocol_entries, None)]

    entries = dict(protocol_entries)
    versions = entries.pop("versions")

    if not isinstance(versions, dict) or not versions:
        msg = (
            f"{where}: 'versions' must be a non-empty mapping of version names "
            f"to (optional) metadata."
        )
        raise ValueError(msg)

    invalid = [str(n) for n, m in versions.items() if not (m is None or isinstance(m, dict))]
    if invalid:
        msg = f"{where}: metadata of versions {invalid} must be a mapping (e.g. {{closing: 0.2}})."
        raise ValueError(msg)

    versions = [{"name": str(n), "metadata": m or {}} for n, m in versions.items()]
    version_names = [v["name"] for v in versions]

    # "." would end up in the protocol class name, which pickle cannot resolve
    dotted = [v for v in version_names if "." in v]
    if dotted:
        msg = (
            f"{where}: version names must not contain '.' (got {dotted}); "
            f"please rename them (e.g. 'v2.1' -> 'v2_1')."
        )
        raise ValueError(msg)

    return [
        (f"{protocol}@{v['name']}", _apply_version(entries, v["name"]), v) for v in versions
    ]


# To ease the understanding of future me, all comments inside Registry codebase
# assume the existence of the following database.yml files.

# ======================================
# Content of /path/to/first/database.yml
# ======================================
# Databases:
#     DatabaseA:
#         - relative/path/A/trn/{uri}.wav
#         - relative/path/A/dev/{uri}.wav
#         - relative/path/A/tst/{uri}.wav
#     DatabaseB: /absolute/path/B/{uri}.wav
#
# Protocols:
#     DatabaseA:
#         SpeakerDiarization:
#             ProtocolA:
#                 train:
#                     uri: relative/path/A/trn.lst
#                 development:
#                     uri: relative/path/A/dev.lst
#                 test:
#                     uri: relative/path/A/tst.lst
#             ProtocolB:
#                 ...
#     DatabaseB:
#         SpeakerDiarization:
#             Protocol:
#                 ...
#     X:
#         SpeakerDiarization:
#             A_and_B:
#                 train: ...
#                 development: ...
#                 test: ...

# ======================================
# Content of /path/to/second/database.yml
# ======================================
# Databases:
#     DatabaseC: /absolute/path/C/{uri}.wav
#     DatabaseB: /absolute/path/B/{uri}.wav
# Protocols:
#     DatabaseB:
#         SpeakerDiarization:
#             Protocol:
#                ...
#     DatabaseC:
#         SpeakerDiarization:
#             Protocol:
#                 ...


class Registry:
    """Database and experimental protocols registry

    Usage
    -----
    >>> from pyannote.database import registry
    >>> registry.load_database("/path/to/first/database.yml")
    >>> registry.load_database("/path/to/second/database.yml")
    """

    def __init__(self) -> None:
        # Mapping of database.yml paths to their config in a dictionary
        # Example after loading both database.yml:
        #   {"/path/to/first/database.yml": {
        #           "Databases":{
        #               "DatabaseA": ["relative/path/A/trn/{uri}.wav", "relative/path/A/dev/{uri}.wav", relative/path/A/tst/{uri}.wav]
        #               "DatabaseB": "/absolute/path/B/{uri}.wav"
        #           },
        #           "Protocols":{
        #               "DatabaseA":{
        #                   "SpeakerDiarization": {
        #                       "ProtocolA": {
        #                           "train": {"uri": "relative/path/A/trn.lst"},
        #                           "development": {"uri": "relative/path/A/dev.lst"},
        #                           "test": {"uri"; "relative/path/A/tst.lst"}
        #                       }
        #                   }
        #               },
        #               "DatabaseB":{"SpeakerDiarization":{"Protocol": {...}}},
        #               "X":{"SpeakerDiarization":{"A_and_B":{...}}}
        #           }
        #       },
        #    "/path/to/second/database.yml": {
        #           "Databases":{
        #               "DatabaseC": /absolute/path/C/{uri}.wav
        #               "DatabaseB": "/absolute/path/B/{uri}.wav"
        #           },
        #           "Protocols":{
        #               "DatabaseB":{"SpeakerDiarization": {"Protocol": {...}}},
        #               "DatabaseC":{...}
        #           }
        #       }
        #   }
        self.configs: Dict[Path, Dict] = dict()

        # Content of the "Database" root item (= where to find file content),
        # with paths as written in database.yml. Only used by FileFinder.
        #   {
        #   "DatabaseA": [
        #       "relative/path/A/trn/{uri}.wav",
        #       "relative/path/A/dev/{uri}.wav",
        #       "relative/path/A/tst/{uri}.wav",
        #       ],
        #   "DatabaseB": ["/absolute/path/B/{uri}.wav"],
        #   "DatabaseC": ["/absolute/path/C/{uri}.wav"]
        #   }
        self.sources: dict[str, list[str]] = dict()

        # Directory of the database.yml defining each database's sources
        # (relative sources are relative to it). Only used by FileFinder.
        # Example after loading both database.yml:
        #   {
        #   "DatabaseA": Path("/path/to/first"),
        #   "DatabaseB": Path("/path/to/second"),
        #   "DatabaseC": Path("/path/to/second"),
        #   }
        self.source_dirs: dict[str, Path] = dict()

        # Mapping of database names to a type that inherits from Database
        # Example after loading both database.yml:
        #   {"DatabaseA": pyannote.database.registry.DatabaseA,
        #    "DatabaseB": pyannote.database.registry.DatabaseB,
        #    "DatabaseC": pyannote.database.registry.DatabaseC,
        #    "X": pyannote.database.registry.X}
        self.databases: Dict[Text, Type] = dict()

    def load_database(
        self,
        path: Union[Text, Path],
        mode: LoadingMode = LoadingMode.OVERRIDE,
    ):
        """Load YAML configuration file into the registry

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.
        mode : LoadingMode, optional
            Controls how to handle conflicts in protocol names.
            Defaults to overriding the existing protocol.

        Usage
        -----
        >>> from pyannote.database import registry
        >>> registry.load_database("/path/to/database.yml")
        """

        self._load_database_helper(path, mode=mode, loading=set())
        self._reload_meta_protocols()

    def _load_database_helper(
        self,
        database_yml: Union[Text, Path],
        mode: LoadingMode = LoadingMode.KEEP,
        loading: Set[Path] = set(),
    ):
        """Helper function for recursive loading

        Parameters
        ----------
        database_yml : Union[Text, Path]
            Path to the database.yml
        mode : LoadingMode, optional
            Controls how to handle conflicts in protocol names.
            Defaults to overriding the existing protocol.
        """

        # make path absolute
        database_yml = Path(database_yml).expanduser().resolve()

        # stop here if configuration file is already being loaded
        # (possibly because of circular requirements)
        if database_yml in loading:
            return

        # mark it as currently being loaded (to avoid future circular requirements)
        loading.add(database_yml)

        # load configuration
        with open(database_yml, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # load every requirement
        requirements = config.pop("Requirements", list())
        if not isinstance(requirements, list):
            requirements = [requirements]
        for requirement_yaml in requirements:
            requirement_yaml = Path(requirement_yaml)
            if not requirement_yaml.is_absolute():
                requirement_yaml = database_yml.parent / requirement_yaml

            self._load_database_helper(requirement_yaml, mode=mode, loading=loading)

        # process "Protocols" section
        protocols = config.get("Protocols", dict())

        # make sure meta-protocols are processed last (relies on the fact that
        # dicts are iterated in insertion order since Python 3.6)
        x = protocols.pop("X", None)
        if x is not None:
            protocols["X"] = x

        # load protocols of each database
        for db_name, db_entries in protocols.items():
            self._load_protocols(db_name, db_entries, database_yml, mode=mode)

        # process "Databases" section
        databases = config.get("Databases", dict())
        for db_name, value in databases.items():
            if not isinstance(value, list):
                value = [value]

            self.sources[str(db_name)] = [str(p) for p in value]
            self.source_dirs[str(db_name)] = database_yml.parent

        # save configuration for later reloading of meta-protocols
        self.configs[database_yml] = config

    def get_database(self, database_name, **kwargs) -> Database:
        """Get database by name

        Parameters
        ----------
        database_name : str
            Database name.

        Returns
        -------
        database : Database
            Database instance
        """

        try:
            database = self.databases[database_name]

        except KeyError:
            if database_name == "X":
                msg = (
                    "Could not find any meta-protocol. Please refer to "
                    "pyannote.database documentation to learn how to define them: "
                    "https://github.com/pyannote/pyannote-database"
                )
            else:
                msg = (
                    'Could not find any protocol for "{name}" database. Please '
                    "refer to pyannote.database documentation to learn how to "
                    "define them: https://github.com/pyannote/pyannote-database"
                )
                msg = msg.format(name=database_name)
            raise ValueError(msg)

        return database(**kwargs)

    def get_protocol(
        self, name, preprocessors: Optional[Preprocessors] = None
    ) -> Protocol:
        """Get protocol by full name

        Parameters
        ----------
        name : str
            Protocol full name (e.g. "Etape.SpeakerDiarization.TV").
            An annotation version may be selected with an "@" suffix
            (e.g. "AMI.SpeakerDiarization.Raw@only_words"); without it,
            the protocol's default version is used.
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

        base_name, at, version = name.partition("@")
        if at and (not version or "@" in version):
            msg = (
                f"Invalid protocol name '{name}': expected at most one '@', "
                f"followed by a version name (e.g. 'Database.Task.Protocol@version')."
            )
            raise ValueError(msg)
        database_name, task_name, protocol_name = base_name.split(".")
        database = self.get_database(database_name)
        try:
            protocol = database.get_protocol(
                task_name,
                f"{protocol_name}@{version}" if version else protocol_name,
                preprocessors=preprocessors,
            )
        except KeyError:
            if not version:
                raise
            available = sorted(
                p.partition("@")[2]
                for p in database.get_protocols(task_name)
                if p.startswith(f"{protocol_name}@")
            )
            msg = (
                f"Could not find version '{version}' of protocol "
                f"'{database_name}.{task_name}.{protocol_name}' "
                f"(available versions: {available or 'none'})."
            )
            raise ValueError(msg)
        protocol.name = name
        return protocol

    # iterate over all protocols by name
    def __iter__(self):
        for database_name in self.databases:
            database = self.get_database(database_name)
            for task_name in database.get_tasks():
                for protocol_name in database.get_protocols(task_name):
                    yield f"{database_name}.{task_name}.{protocol_name}"

    def _load_protocols(
        self,
        db_name,
        db_entries: dict,
        database_yml: Union[Text, Path] = None,
        mode: LoadingMode = LoadingMode.OVERRIDE,
    ):
        """Load all protocols from this database into the registry.

        Parameters
        ----------
        db_name : _type_
            Name of the database
        db_entries : dict
            Dict of all entries under this database (this should be tasks)
        database_yml : Union[Text, Path], optional
            Path to the database.yml file. Not required for X protocols, by default None
        """

        db_name = str(db_name)

        # maps tuple (task,protocol) to the custom protocol class
        protocols: Dict[Tuple[Text, Text], Type] = dict()

        for task_name, task_entries in db_entries.items():
            for protocol, protocol_entries in task_entries.items():
                protocol = str(protocol)

                if "@" in protocol:
                    msg = (
                        f"Protocol name '{protocol}' ({db_name}.{task_name}, "
                        f"{database_yml}) must not contain '@': it is reserved to "
                        f"select an annotation version (e.g. 'Protocol@version')."
                    )
                    raise ValueError(msg)

                # one protocol per version (just `protocol` when unversioned)
                expanded = _expand_versions(
                    db_name, task_name, protocol, protocol_entries, database_yml
                )
                for versioned_protocol, entries, version in expanded:
                    CustomProtocol = create_protocol(
                        db_name,
                        task_name,
                        versioned_protocol,
                        entries,
                        database_yml,
                        version=version,
                    )
                    if CustomProtocol is None:
                        continue

                    protocols[(task_name, versioned_protocol)] = CustomProtocol

                # expanded[0] is the default (first listed) version, e.g. "Raw@original":
                default_protocol, _, default_version = expanded[0]
                if default_version is not None and (task_name, default_protocol) in protocols:
                    protocols[(task_name, protocol)] = protocols[(task_name, default_protocol)]

        # If needed, merge old protocols dict with the new one (according to current override rules)
        if db_name in self.databases:
            old_protocols = self.databases[db_name]._protocols
            _merge_protocols_inplace(
                protocols, old_protocols, mode, db_name, database_yml
            )

        # create database class on-the-fly
        protocol_list = [
            (task, p_name, p_type) for (task, p_name), p_type in protocols.items()
        ]
        self.databases[db_name] = type(
            db_name,
            (Database,),
            {"__init__": get_init(protocol_list), "_protocols": protocols},
        )

    def _reload_meta_protocols(self):
        """Reloads all meta protocols from all database.yml files loaded."""

        # TODO: decide how to handle X protocol overriding.

        self.databases.pop("X", None)

        for db_yml, config in self.configs.items():
            databases = config.get("Protocols", dict())
            if "X" in databases:
                self._load_protocols(
                    "X", databases["X"], db_yml, mode=LoadingMode.OVERRIDE
                )


def _env_config_paths() -> List[Path]:
    """Parse PYANNOTE_DATABASE_CONFIG environment variable

    PYANNOTE_DATABASE_CONFIG may contain multiple paths separation by ";".

    Returns
    -------
    paths : list of Path
        List of all YAML database file defined in PYANNOTE_DATABASE_CONF
    """

    content = os.environ.get("PYANNOTE_DATABASE_CONFIG", "")

    paths = []
    for path in content.split(";"):
        path = Path(path).expanduser()
        if path.is_file():
            paths.append(path)
    return paths


def _find_default_ymls() -> List[Path]:
    """Get paths to default YAML configuration files

    * $HOME/.pyannote/database.yml
    * $CWD/database.yml
    * PYANNOTE_DATABASE_CONFIG environment variable

    Returns
    -------
    paths : list of Path
        List of existing default YAML configuration files
    """

    paths: List[Path] = []

    home_db_yml = Path("~/.pyannote/database.yml").expanduser()
    if home_db_yml.is_file():
        paths.append(home_db_yml)

    cwd_db_yml = Path.cwd() / "database.yml"
    if cwd_db_yml.is_file():
        paths.append(cwd_db_yml)

    paths += _env_config_paths()

    return paths


def _merge_protocols_inplace(
    new_protocols: Dict[Tuple[Text, Text], Type],
    old_protocols: Dict[Tuple[Text, Text], Type],
    mode: LoadingMode,
    db_name: str,
    database_yml: str,
):
    """Merge new and old protocols inplace into the passed new_protocol.

    Warning, merging order might be counterintuitive : "KEEP" strategy keeps element from the OLD protocol
    and MODIFIES the new protocol.

    TODO: make it intuitive :)

    Parameters
    ----------
    new_protocols : Dict[Tuple[Text, Text], Type]
        New protocol dict
        Maps (task,protocol) tuples to custom protocol classes
    old_protocols : Dict[Tuple[Text, Text], Type]
        Old protocols dict
        Maps (task,protocol) tuples to custom protocol classes.
    mode : LoadingMode
        How to handle override
    db_name : _type_
        Name of the database (for logging/warning purposes)
    database_yml : str
        Path of the database.yml file (for logging/warning purposes)
    """

    # for all previously defined protocol (in old_protocols)
    for p_id, old_p in old_protocols.items():
        # if this protocol is redefined
        if p_id in new_protocols:
            t_name, p_name = p_id
            realname = f"{db_name}.{t_name}.{p_name}"

            # raise an error
            if mode == LoadingMode.ERROR:
                raise RuntimeError(
                    f"Cannot load {realname} protocol from '{database_yml}' as it already exists."
                )

            # keep the new protocol
            elif mode == LoadingMode.OVERRIDE:
                warnings.warn(
                    f"Replacing existing {realname} protocol by the one defined in '{database_yml}'."
                )
                pass

            # keep the old protocol
            elif mode == LoadingMode.KEEP:
                warnings.warn(
                    f"Skipping {realname} protocol defined in '{database_yml}' as it already exists."
                )
                new_protocols[p_id] = old_p

        # no conflit : keep the previously defined protocol
        else:
            new_protocols[p_id] = old_p


# initialize the registry singleton
registry = Registry()

# load all database yaml files found at startup
for yml in _find_default_ymls():
    registry.load_database(yml)
