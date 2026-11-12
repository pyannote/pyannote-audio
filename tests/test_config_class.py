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


def test_get_pipeline_ignores_nested_trust_flag(monkeypatch):
    """A config cannot escalate trust by passing trust_remote_code in its dict.

    Load getter.py from the source tree with lightweight stubs so this test
    does not require a full pyannote.audio install.
    """
    import importlib.util
    import sys
    import types

    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return object()

    stubs = {
        "torch": types.ModuleType("torch"),
        "torch_audiomentations": types.ModuleType("torch_audiomentations"),
        "torch_audiomentations.core": types.ModuleType("torch_audiomentations.core"),
        "torch_audiomentations.core.transforms_interface": types.ModuleType(
            "torch_audiomentations.core.transforms_interface"
        ),
        "torch_audiomentations.utils": types.ModuleType("torch_audiomentations.utils"),
        "torch_audiomentations.utils.config": types.ModuleType(
            "torch_audiomentations.utils.config"
        ),
        "pyannote": types.ModuleType("pyannote"),
        "pyannote.audio": types.ModuleType("pyannote.audio"),
        "pyannote.audio.core": types.ModuleType("pyannote.audio.core"),
        "pyannote.audio.core.calibration": types.ModuleType(
            "pyannote.audio.core.calibration"
        ),
        "pyannote.audio.core.model": types.ModuleType("pyannote.audio.core.model"),
        "pyannote.audio.core.pipeline": types.ModuleType("pyannote.audio.core.pipeline"),
        "pyannote.audio.core.plda": types.ModuleType("pyannote.audio.core.plda"),
    }
    stubs["torch_audiomentations.core.transforms_interface"].BaseWaveformTransform = object
    stubs["torch_audiomentations.utils.config"].from_dict = lambda *a, **k: None
    stubs["pyannote.audio.core.calibration"].Calibration = object
    stubs["pyannote.audio.core.model"].Model = object
    stubs["pyannote.audio.core.pipeline"].Pipeline = FakePipeline
    stubs["pyannote.audio.core.plda"].PLDA = object

    added = []
    for name, module in stubs.items():
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, module)
            added.append(name)

    getter_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pyannote"
        / "audio"
        / "pipelines"
        / "utils"
        / "getter.py"
    )
    spec = importlib.util.spec_from_file_location("pyannote_getter_under_test", getter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.get_pipeline({"checkpoint": "untrusted/repo", "trust_remote_code": True})
    assert captured["kwargs"]["trust_remote_code"] is False

    captured.clear()
    module.get_pipeline(
        {"checkpoint": "untrusted/repo", "trust_remote_code": True},
        trust_remote_code=True,
    )
    assert captured["kwargs"]["trust_remote_code"] is True
