# Copyright (c) 2025- pyannoteAI

import hashlib
import io
import json
import tarfile
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

from scripts.release import NAMES, check_pypi, validate_distributions


def distributions(directory, versions=None, requires=None):
    for name in NAMES:
        version = (versions or {}).get(name, "7.0.0")
        basename = f"{name.replace('-', '_')}-{version}"
        metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
        if requires:
            metadata += f"Requires-Dist: {requires}\n"
        with ZipFile(directory / f"{basename}-py3-none-any.whl", "w") as wheel:
            wheel.writestr(f"{basename}.dist-info/METADATA", metadata)
        with tarfile.open(directory / f"{basename}.tar.gz", "w:gz") as source:
            data = metadata.encode()
            info = tarfile.TarInfo(f"{basename}/PKG-INFO")
            info.size = len(data)
            source.addfile(info, io.BytesIO(data))


def test_complete_release(tmp_path):
    distributions(tmp_path)
    version, artifacts = validate_distributions(tmp_path, "v7.0.0")
    assert str(version) == "7.0.0"
    assert len(artifacts) == 10


def test_missing_package(tmp_path):
    distributions(tmp_path)
    next(tmp_path.glob("pyannote_audio-*.whl")).unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_distributions(tmp_path)


def test_mixed_versions(tmp_path):
    distributions(tmp_path, {"pyannote-core": "7.0.1"})
    with pytest.raises(ValueError, match="share one version"):
        validate_distributions(tmp_path)


def test_wrong_tag(tmp_path):
    distributions(tmp_path)
    with pytest.raises(ValueError, match="tag"):
        validate_distributions(tmp_path, "7.0.1")


def test_pre_monorepo_version(tmp_path):
    distributions(tmp_path, dict.fromkeys(NAMES, "6.2.0"))
    with pytest.raises(ValueError, match="start at 7.0.0"):
        validate_distributions(tmp_path)


def test_incompatible_internal_dependency(tmp_path):
    distributions(tmp_path, requires="pyannote-core<7")
    with pytest.raises(ValueError, match="rejects shared version"):
        validate_distributions(tmp_path)


@pytest.mark.parametrize("status", [404, 403, 500])
def test_pypi_errors(tmp_path, monkeypatch, status):
    distributions(tmp_path)
    version, artifacts = validate_distributions(tmp_path)

    def fail(*args, **kwargs):
        raise HTTPError("url", status, "error", {}, None)

    monkeypatch.setattr("scripts.release.urlopen", fail)
    if status == 404:
        check_pypi(version, artifacts)
    else:
        with pytest.raises(HTTPError):
            check_pypi(version, artifacts)


@pytest.mark.parametrize("conflict", [False, True])
def test_partial_publication_retry(tmp_path, monkeypatch, conflict):
    distributions(tmp_path)
    version, artifacts = validate_distributions(tmp_path)
    artifact = artifacts["pyannote-core", "wheel"]
    published = {
        "filename": artifact.name,
        "digests": {
            "sha256": "wrong"
            if conflict
            else hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
    }

    def response(url, **kwargs):
        return io.BytesIO(
            json.dumps(
                {"urls": [published] if "/pyannote-core/" in url else []}
            ).encode()
        )

    monkeypatch.setattr("scripts.release.urlopen", response)
    if conflict:
        with pytest.raises(ValueError, match="different contents"):
            check_pypi(version, artifacts)
    else:
        check_pypi(version, artifacts)
