#!/usr/bin/env python
# encoding: utf-8

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

import glob
import os
import warnings
from pathlib import Path
from typing import Text

from pyannote.database.protocol.protocol import ProtocolFile

from .registry import Registry
from .registry import registry as global_registry


def _scan_candidate(path: Path) -> list[Path]:
    """Find files matching a path

    Parameters
    ----------
    path : Path
        Path to a file, possibly with "*" or "**" glob patterns.

    Returns
    -------
    matches : list of Path
        Existing files matching `path`.
    """

    if "*" not in str(path):
        return [path] if path.is_file() else []

    pattern = glob.escape(str(path)).replace("[*]", "*")
    return [p for match in glob.iglob(pattern, recursive=True) if (p := Path(match)).is_file()]


class FileFinder:
    """Database file finder. 
    
    Retrieve media files by URI.

    Parameters
    ----------
    registry : Registry, optional
        Database registry. Defaults to `pyannote.database.registry`.
    """

    def __init__(
        self, 
        registry: Registry = None,
        database_yml: Text = None):
        super().__init__()
        if registry is None:
            if database_yml is None:
                registry = global_registry
            else:
                warnings.warn("Passing `database.yml` to `FileFinder` is deprecated in favor of `registry`.")
                registry = Registry()
                registry.load_database(database_yml)
        self.registry = registry

    def __call__(self, current_file: ProtocolFile) -> Path:
        """Look for current file

        Parameter
        ---------
        current_file : ProtocolFile
            Protocol file.

        Returns
        -------
        path : Path
            Path to file.

        Raises
        ------
        FileNotFoundError when the file could not be found or when more than one
        matching file were found.
        """

        uri = current_file["uri"]
        database = current_file["database"]

        content = os.environ.get("PYANNOTE_DATABASE_DATA_DIRS", "")
        data_dirs = [Path(d).expanduser() for d in content.split(os.pathsep) if d]
        data_dirs = list(dict.fromkeys(data_dirs))  # Remove duplicates
        yml_dir = self.registry.source_dirs[database]

        searched: list[Path] = []
        fallback: list[Path] = []
        for template in self.registry.sources[database]:
            path = Path(template.format(uri=uri, database=database))
            if path.is_absolute():
                searched.append(path)
            elif data_dirs:
                searched.extend(data_dir / path for data_dir in data_dirs)
                fallback.append(yml_dir / path)
            else:
                searched.append(yml_dir / path)

        found = [match for path in searched for match in _scan_candidate(path)]

        # relative paths are looked for next to database.yml only as a last resort
        if not found and fallback:
            searched.extend(fallback)
            found = [match for path in fallback for match in _scan_candidate(path)]

        if len(found) == 1:
            return found[0]

        if len(found) == 0:
            msg = f'Could not find file "{uri}" in any of the following location(s):'
            for path in searched:
                msg += f"\n - {path}"
            raise FileNotFoundError(msg)

        if len(found) > 1:
            msg = (
                f'Looked for file "{uri}" and found more than one '
                f"({len(found)}) matching locations: "
            )
            for path in found:
                msg += f"\n - {path}"
            raise FileNotFoundError(msg)
