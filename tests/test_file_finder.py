"""PYANNOTE_DATABASE_DATA_DIRS tests

Relative `Databases:` paths are searched in every `PYANNOTE_DATABASE_DATA_DIRS`
directory (duplicates ignored); a file found in more than one raises
`FileNotFoundError`. Only if none contains the file is it searched next to
`database.yml`. Absolute paths are used as is.
"""

import os

import pytest

from pyannote.database.file_finder import FileFinder
from pyannote.database.registry import Registry


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Each file name tells where the file exists:

    yml/   database.yml, only_yml.wav, yml_and_a.wav
    a/     only_a.wav, yml_and_a.wav, a_and_b.wav
    b/     only_b.wav, a_and_b.wav
    c[1]?/ e[1].wav, e1.wav, f]1.wav, what?.wav, whatX.wav, sub/nested.wav,
           folder.wav/ (a directory) and sub/folder.wav
    abs/   absolute.wav (declared with an absolute path)
    """
    root = tmp_path_factory.mktemp("file_finder")
    for directory, uris in {
        "yml": ["only_yml", "yml_and_a"],
        "a": ["only_a", "yml_and_a", "a_and_b"],
        "b": ["only_b", "a_and_b"],
        "c[1]?": ["e[1]", "e1", "f]1", "what?", "whatX", "sub/nested", "sub/folder"],
    }.items():
        (root / directory / "audio" / "sub").mkdir(parents=True)
        for uri in uris:
            (root / directory / "audio" / f"{uri}.wav").touch()
    (root / "c[1]?" / "audio" / "folder.wav").mkdir()
    (root / "abs").mkdir()
    (root / "abs" / "absolute.wav").touch()

    (root / "yml" / "database.yml").write_text(
        "Databases:\n"
        "  Relative: audio/{uri}.wav\n"
        "  Glob: audio/**/{uri}.wav\n"
        f"  Absolute: {root}/abs/{{uri}}.wav\n"
    )
    return root


def _find(root, monkeypatch, data_dirs, uri, database="Relative"):
    if data_dirs is None:
        monkeypatch.delenv("PYANNOTE_DATABASE_DATA_DIRS", raising=False)
    else:
        data_dirs = os.pathsep.join(str(root / d) for d in data_dirs)
        monkeypatch.setenv("PYANNOTE_DATABASE_DATA_DIRS", data_dirs)
    registry = Registry()
    registry.load_database(root / "yml" / "database.yml")
    return FileFinder(registry=registry)({"uri": uri, "database": database})


@pytest.mark.parametrize(
    "data_dirs, database, uri, expected",
    [
        # PYANNOTE_DATABASE_DATA_DIRS not set: next to database.yml
        (None, "Relative", "only_yml", "yml/audio/only_yml.wav"),
        # searched in every data directory
        (["a", "b"], "Relative", "only_a", "a/audio/only_a.wav"),
        (["a", "b"], "Relative", "only_b", "b/audio/only_b.wav"),
        # next to database.yml only if no data directory has the file
        (["a", "b"], "Relative", "only_yml", "yml/audio/only_yml.wav"),
        (["a", "b"], "Relative", "yml_and_a", "a/audio/yml_and_a.wav"),
        # duplicate data directories are ignored
        (["a", "a"], "Relative", "only_a", "a/audio/only_a.wav"),
        # absolute paths are not affected
        (None, "Absolute", "absolute", "abs/absolute.wav"),
        (["a", "b"], "Absolute", "absolute", "abs/absolute.wav"),
        # "?", "[" and "]" are matched literally (in uris and data directories)
        (["c[1]?"], "Glob", "f]1", "c[1]?/audio/f]1.wav"),
        (["c[1]?"], "Glob", "what?", "c[1]?/audio/what?.wav"),
        # "**" also searches subdirectories, and only matches files
        (["c[1]?"], "Glob", "nested", "c[1]?/audio/sub/nested.wav"),
        (["c[1]?"], "Glob", "folder", "c[1]?/audio/sub/folder.wav"),
    ],
)
def test_file_is_found(root, monkeypatch, data_dirs, database, uri, expected):
    assert _find(root, monkeypatch, data_dirs, uri, database) == root / expected


@pytest.mark.parametrize(
    "uri, match, listed, not_listed",
    [
        # found in more than one data directory (database.yml not searched)
        ("a_and_b", "more than one", ["a", "b"], ["yml"]),
        # found nowhere: every searched location is listed
        ("missing", "Could not find", ["a", "b", "yml"], []),
    ],
)
def test_file_is_not_found(root, monkeypatch, uri, match, listed, not_listed):
    with pytest.raises(FileNotFoundError, match=match) as error:
        _find(root, monkeypatch, ["a", "b"], uri)
    for directory in listed:
        assert str(root / directory / "audio" / f"{uri}.wav") in str(error.value)
    for directory in not_listed:
        assert str(root / directory) not in str(error.value)
