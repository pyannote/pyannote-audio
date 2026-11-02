"""Annotation versioning tests

`versions:` is declared per protocol; a protocol is versioned if (and only
if) it declares `versions:`, and `{version}` is substituted wherever it
appears (paths without it are shared by all versions). Different
protocols (even in the same task) may declare different version sets. The
default version (no "@version" requested) is always the first one listed.
"""

import pytest

from pyannote.database import registry as global_registry
from pyannote.database.registry import Registry


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _duration(file):
    return file["annotation"].get_timeline().duration()


def _protocols(database):
    return set(global_registry.get_database(database).get_protocols("SpeakerDiarization"))


@pytest.fixture(scope="module")
def versioned_dataset(tmp_path_factory):
    """Both versions cover the same recordings (u1 + u2, shared uri list).

    - Raw: [original, refined] -> 1.0 s / 2.0 s segments
    - Other: [refined] only -> different version set, default is not "original"
    - Shared: [original, refined] but no {version} in its paths
    - Flat: no `versions:` -> stays unversioned
    """
    root = tmp_path_factory.mktemp("versioned")

    _write(root / "lists" / "train.lst", "u1\nu2\n")

    for uri in ("u1", "u2"):
        for version, duration in (("original", 1.0), ("refined", 2.0)):
            _write(
                root / "rttm" / version / f"{uri}.rttm",
                f"SPEAKER {uri} 1 0.0 {duration} <NA> <NA> spk_{uri} <NA> <NA>\n",
            )
        _write(root / "uem" / f"{uri}.uem", f"{uri} 1 0.000 10.000\n")

    _write(
        root / "database.yml",
        """
Protocols:
  VersionedTestDB:
    SpeakerDiarization:
      Raw:
        versions:                         # default is "original" (first listed)
          original:
          refined:
            closing: 0.2
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/{version}/{uri}.rttm
          annotated: uem/{uri}.uem
      Other:
        versions: {refined: {closing: 0.2}}
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/{version}/{uri}.rttm
      Shared:
        versions: {original, refined}     # no {version}: same files for both
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/original/{uri}.rttm
      Flat:                               # no versions: -> stays unversioned
        scope: file
        train:
          uri: lists/train.lst
          annotation: rttm/original/{uri}.rttm
  X:
    SpeakerDiarization:
      VersionedTestCompound:
        train:
          VersionedTestDB.SpeakerDiarization.Raw@refined: [train]
""",
    )

    # the compound (X) member resolution goes through the global registry
    global_registry.load_database(root / "database.yml")
    return root


def test_selecting_a_version_changes_annotation_only(versioned_dataset):
    """Selecting a version changes the annotation content, not which
    recordings are covered."""
    original = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@original")
    refined = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Raw@refined")

    assert sorted(f["uri"] for f in original.train()) == ["u1", "u2"]
    assert sorted(f["uri"] for f in refined.train()) == ["u1", "u2"]

    original_u1 = next(f for f in original.train() if f["uri"] == "u1")
    refined_u1 = next(f for f in refined.train() if f["uri"] == "u1")
    assert _duration(original_u1) == 1.0
    assert _duration(refined_u1) == 2.0
    assert original_u1["version"] == {"name": "original", "metadata": {}}
    assert refined_u1["version"] == {"name": "refined", "metadata": {"closing": 0.2}}


def test_one_protocol_per_version_and_default_is_first_listed(versioned_dataset):
    """Every version is registered as "Protocol@version", plus the bare name
    for the first version listed. Sibling protocols have their own version sets."""
    protocols = _protocols("VersionedTestDB")
    assert {"Raw", "Raw@original", "Raw@refined", "Other", "Other@refined"} <= protocols
    assert "Other@original" not in protocols

    for name, default in (
        ("Raw", {"name": "original", "metadata": {}}),
        ("Other", {"name": "refined", "metadata": {"closing": 0.2}}),
    ):
        files = list(global_registry.get_protocol(f"VersionedTestDB.SpeakerDiarization.{name}").train())
        assert sorted(f["uri"] for f in files) == ["u1", "u2"]
        assert all(f["version"] == default for f in files)


def test_versions_without_placeholder_are_still_expanded(versioned_dataset):
    """Declaring `versions:` is enough to version a protocol: paths without
    `{version}` are simply shared by every version."""
    assert {"Shared", "Shared@original", "Shared@refined"} <= _protocols("VersionedTestDB")

    files = list(global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Shared@refined").train())
    assert all(f["version"]["name"] == "refined" and _duration(f) == 1.0 for f in files)


def test_unversioned_protocol_stays_flat(versioned_dataset):
    """A protocol with no `versions:` is not expanded, even when a sibling
    protocol in the same task is versioned."""
    protocols = _protocols("VersionedTestDB")
    assert "Flat" in protocols
    assert not any(p.startswith("Flat@") for p in protocols)

    flat = global_registry.get_protocol("VersionedTestDB.SpeakerDiarization.Flat")
    for file in flat.train():
        # iterate keys: does not trigger lazy evaluation
        assert "version" not in set(file)


def test_compound_member_with_version(versioned_dataset):
    """X meta-protocol members can pin an annotation version with @."""
    compound = global_registry.get_protocol("X.SpeakerDiarization.VersionedTestCompound")
    files = list(compound.train())

    assert sorted(f["uri"] for f in files) == ["u1", "u2"]
    assert all(f["version"]["name"] == "refined" for f in files)


@pytest.mark.parametrize(
    "name, match",
    [
        ("Raw@nope", r"version 'nope'.*available versions.*original"),
        ("Raw@", r"Invalid protocol name.*at most one '@'"),
        ("Raw@original@refined", r"Invalid protocol name.*at most one '@'"),
    ],
)
def test_invalid_version_selection_is_rejected(versioned_dataset, name, match):
    """Unknown versions, an empty "@" suffix or several "@" are clear errors."""
    with pytest.raises(ValueError, match=match):
        global_registry.get_protocol(f"VersionedTestDB.SpeakerDiarization.{name}")


@pytest.mark.parametrize(
    "protocol_block, match",
    [
        # "version" is a reserved subset entry, even without `versions:`
        ("Raw:\n  train:\n    version: not_allowed", r"reserved for annotation versioning"),
        # "@" is reserved to select a version
        ("Foo@Bar:\n  scope: file", r"Foo@Bar.*must not contain '@'"),
        # versions map names to (optional) metadata mappings
        ("Raw:\n  versions: [original, refined]", r"must be a non-empty mapping"),
        ("Raw:\n  versions: {}", r"must be a non-empty mapping"),
        ("Raw:\n  versions: {original, refined: 0.2}", r"metadata of versions \['refined'\]"),
        ("Raw:\n  versions: {original, refined: 0}", r"metadata of versions \['refined'\]"),
        # "." would make the protocol class unpicklable
        ("Raw:\n  versions: {original, v2.1}", r"must not contain '\.'.*v2\.1"),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, protocol_block, match):
    block = "\n".join(f"      {line}" for line in protocol_block.splitlines())
    _write(
        tmp_path / "database.yml",
        f"Protocols:\n  InvalidDB:\n    SpeakerDiarization:\n{block}\n",
    )
    with pytest.raises(ValueError, match=match):
        Registry().load_database(tmp_path / "database.yml")
