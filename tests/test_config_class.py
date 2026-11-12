import importlib.util
from pathlib import Path

import pytest

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "pyannote"
    / "audio"
    / "utils"
    / "config_class.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("pyannote_config_class", _HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_and_unqualified_names_are_allowed():
    helper = _load()
    assert (
        helper.assert_trusted_config_class(
            "pyannote.audio.pipelines.SpeakerDiarization",
            "pyannote.pipeline.blocks",
        )
        == "pyannote.audio.pipelines.SpeakerDiarization"
    )
    assert (
        helper.assert_trusted_config_class(
            "FileFinder",
            "pyannote.audio",
        )
        == "pyannote.audio.FileFinder"
    )


def test_untrusted_class_names_are_rejected():
    helper = _load()
    with pytest.raises(ValueError, match="Refusing to import"):
        helper.assert_trusted_config_class("os.system", "pyannote.audio")
    with pytest.raises(ValueError, match="Refusing to import"):
        helper.assert_trusted_config_class(
            "subprocess.Popen", "pyannote.pipeline.blocks"
        )


def test_trust_remote_code_bypasses_allowlist():
    helper = _load()
    assert (
        helper.assert_trusted_config_class(
            "my_pkg.CustomPipeline",
            "pyannote.audio",
            trust_remote_code=True,
        )
        == "my_pkg.CustomPipeline"
    )
