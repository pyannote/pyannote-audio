# Copyright (c) 2025- pyannoteAI
"""Validate a complete, synchronized set of distributions before publication."""

import argparse
import hashlib
import json
import tarfile
from email.parser import BytesParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

from scripts._common import PACKAGES

NAMES = {f"pyannote-{name}" for name in PACKAGES}


def validate_distributions(directory: Path, tag: str | None = None):
    artifacts = {}
    versions = set()
    metadata_by_name = {}
    for path in sorted([*directory.glob("*.whl"), *directory.glob("*.tar.gz")]):
        if path.suffix == ".whl":
            kind = "wheel"
            filename_name, filename_version, _, _ = parse_wheel_filename(path.name)
            with ZipFile(path) as archive:
                metadata_paths = [
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                ]
                if len(metadata_paths) != 1:
                    raise ValueError(f"Expected one METADATA in {path.name}")
                content = archive.read(metadata_paths[0])
        else:
            kind = "sdist"
            filename_name, filename_version = parse_sdist_filename(path.name)
            with tarfile.open(path) as archive:
                metadata_paths = [
                    n
                    for n in archive.getnames()
                    if n.count("/") == 1 and n.endswith("/PKG-INFO")
                ]
                if len(metadata_paths) != 1:
                    raise ValueError(f"Expected one PKG-INFO in {path.name}")
                content = archive.extractfile(metadata_paths[0]).read()
        metadata = BytesParser().parsebytes(content)
        name = canonicalize_name(metadata["Name"])
        version = Version(metadata["Version"])
        if name not in NAMES or name != filename_name or version != filename_version:
            raise ValueError(
                f"Unexpected package or inconsistent filename/metadata: {path.name}"
            )
        key = (name, kind)
        if key in artifacts:
            raise ValueError(f"Duplicate distribution for {name}: {kind}")
        artifacts[key] = path
        versions.add(version)
        metadata_by_name[key] = metadata
    expected = {(name, kind) for name in NAMES for kind in ("wheel", "sdist")}
    if artifacts.keys() != expected:
        raise ValueError(
            f"Expected one wheel and sdist for every package; missing {expected - artifacts.keys()}"
        )
    if len(versions) != 1:
        raise ValueError(f"All five packages must share one version: {versions}")
    version = versions.pop()
    if version < Version("7.0.0.dev0"):
        raise ValueError("Shared releases must start at 7.0.0")
    if tag is not None:
        if (
            str(version) != tag.removeprefix("v")
            or version.is_devrelease
            or version.local
        ):
            raise ValueError(
                f"Release tag {tag!r} does not match publishable version {version}"
            )
    for (name, kind), metadata in metadata_by_name.items():
        for dependency in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(dependency)
            if canonicalize_name(
                requirement.name
            ) in NAMES and not requirement.specifier.contains(
                version, prereleases=True
            ):
                raise ValueError(
                    f"{name} {kind}: {requirement} rejects shared version {version}"
                )
    return version, artifacts


def check_pypi(version, artifacts):
    """Permit unpublished or byte-identical files, and reject conflicting uploads."""
    for name in sorted(NAMES):
        try:
            with urlopen(
                f"https://pypi.org/pypi/{name}/{version}/json", timeout=30
            ) as response:
                published = json.load(response)["urls"]
        except HTTPError as error:
            if error.code != 404:
                raise
            published = []
        hashes = {item["filename"]: item["digests"]["sha256"] for item in published}
        for kind in ("wheel", "sdist"):
            artifact = artifacts[name, kind]
            if (
                artifact.name in hashes
                and hashlib.sha256(artifact.read_bytes()).hexdigest()
                != hashes[artifact.name]
            ):
                raise ValueError(
                    f"{artifact.name} already exists on PyPI with different contents"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--tag")
    parser.add_argument("--check-pypi", action="store_true")
    args = parser.parse_args()
    version, artifacts = validate_distributions(args.dist, args.tag)
    if args.check_pypi:
        check_pypi(version, artifacts)
    print(f"Validated {len(artifacts)} distributions for shared version {version}")


if __name__ == "__main__":
    main()
