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

import importlib.metadata
from importlib.metadata import PackageNotFoundError

from packaging.version import InvalidVersion, Version


class MissingDependency(Exception):
    """Exception raised when a required dependency is missing."""

    def __init__(self, what: str, dependency: str, required: Version) -> None:
        super().__init__(
            f"{what} requires {dependency} ~ {required} but it is not installed."
        )
        self.dependency = dependency
        self.required = required


class WrongDependencyVersion(Exception):
    """Exception raised when a required dependency has an invalid version."""

    def __init__(
        self, what: str, dependency: str, required: Version, available: Version
    ) -> None:
        super().__init__(
            f"{what} requires {dependency} ~ {required} but {available} is installed."
        )
        self.dependency = dependency
        self.required = required
        self.available = available


def check_dependencies(dependencies: dict[str, str], what: str) -> None:
    """Check if required dependencies are installed

    Raises
    ------
    MissingDependency
        If a required dependency is not found.
    WrongDependencyVersion
        If a required dependency has an incompatible version.
    """

    for dependency, version in dependencies.items():
        required: Version = Version(version)

        try:
            _version = importlib.metadata.version(dependency)

        except PackageNotFoundError:
            raise MissingDependency(what, dependency, required)

        _version = importlib.metadata.version(dependency)

        try:
            available: Version = Version(_version)

        except InvalidVersion:
            # if the version cannot be parsed, we take our chance and assume it is compatible
            continue

        if available.major != required.major:
            # before pyannote.audio 4.x, we were not strict about the version. therefore,
            # we take our chance and let it fails later in the code (if it ever fails)
            if dependency == "pyannote.audio" and required.major < 4:
                continue

            else:
                # for all other dependencies, we raise an error on major version mismatch
                raise WrongDependencyVersion(what, dependency, required, available)
