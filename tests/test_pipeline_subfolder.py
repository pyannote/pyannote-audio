# MIT License
#
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

"""Tests for subfolder support in Pipeline.from_pretrained"""

import pytest
import yaml
from pathlib import Path

from pyannote.audio.core.pipeline import Pipeline, expand_subfolders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_config(directory: Path, config: dict, subfolder: str | None = None) -> Path:
    """Write a config.yaml inside *directory* (or *directory/subfolder*)."""
    target = directory / subfolder if subfolder else directory
    target.mkdir(parents=True, exist_ok=True)
    cfg_path = target / "config.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


# from_pretrained always injects `token` and `cache_dir` into params via
# setdefault() before calling Klass(**params), so the subclass __init__ must
# accept (and ignore) them.
class _DummyPipeline(Pipeline):
    """Minimal concrete subclass of Pipeline for testing from_pretrained."""

    def __init__(self, token=None, cache_dir=None, **kwargs):
        super().__init__()

    def apply(self, file, **kwargs):
        return None


_DUMMY_CLS_DOTPATH = f"{_DummyPipeline.__module__}.{_DummyPipeline.__qualname__}"


class _DummyParentPipeline(Pipeline):
    """Pipeline that owns a single subpipeline loaded from a checkpoint dict."""

    def __init__(self, sub=None, token=None, cache_dir=None):
        super().__init__()
        if isinstance(sub, dict):
            # Assigning a Pipeline instance triggers __setattr__ which registers
            # it in self._pipelines — the standard pyannote sub-pipeline pattern.
            self.sub = Pipeline.from_pretrained(**sub)

    def apply(self, file, **kwargs):
        return None


_DUMMY_PARENT_CLS_DOTPATH = (
    f"{_DummyParentPipeline.__module__}.{_DummyParentPipeline.__qualname__}"
)


# ---------------------------------------------------------------------------
# expand_subfolders unit tests
# ---------------------------------------------------------------------------


class TestExpandSubfolders:
    """expand_subfolders resolves $model/... references in config dicts/lists."""

    def test_no_model_references(self):
        config = {"key": "plain_value", "nested": {"k": 42}}
        expand_subfolders(config, model_id="org/repo")
        assert config == {"key": "plain_value", "nested": {"k": 42}}

    def test_dict_model_reference_no_parent_subfolder(self):
        config = {"embedding": "$model/embeddings"}
        expand_subfolders(config, model_id="org/repo", token="tok")
        assert config["embedding"] == {
            "checkpoint": "org/repo",
            "revision": None,
            "subfolder": "embeddings",
            "token": "tok",
            "cache_dir": None,
        }

    def test_dict_model_reference_with_parent_subfolder(self):
        """Child $model refs are resolved under parent_subfolder."""
        config = {"segmentation": "$model/seg"}
        expand_subfolders(
            config,
            model_id="org/repo",
            parent_subfolder="v1",
            token=None,
        )
        assert config["segmentation"]["subfolder"] == "v1/seg"

    def test_dict_model_reference_with_explicit_revision(self):
        """@revision in $model/path@rev overrides parent_revision."""
        config = {"model": "$model/weights@abc123"}
        expand_subfolders(config, model_id="org/repo", parent_revision="main")
        assert config["model"]["revision"] == "abc123"
        assert config["model"]["subfolder"] == "weights"

    def test_dict_model_reference_inherits_parent_revision(self):
        config = {"model": "$model/weights"}
        expand_subfolders(config, model_id="org/repo", parent_revision="v2")
        assert config["model"]["revision"] == "v2"

    def test_list_model_reference_no_parent_subfolder(self):
        config = ["$model/part_a", "$model/part_b"]
        expand_subfolders(config, model_id="org/repo")
        assert config[0]["subfolder"] == "part_a"
        assert config[1]["subfolder"] == "part_b"

    def test_list_model_reference_with_parent_subfolder(self):
        config = ["$model/seg", "plain"]
        expand_subfolders(config, model_id="org/repo", parent_subfolder="root")
        assert config[0]["subfolder"] == "root/seg"
        assert config[1] == "plain"

    def test_nested_dict_model_references(self):
        """Recursion into nested dicts."""
        config = {"outer": {"inner": "$model/deep"}}
        expand_subfolders(config, model_id="org/repo", parent_subfolder="base")
        assert config["outer"]["inner"]["subfolder"] == "base/deep"

    def test_nested_list_model_references(self):
        """Recursion into lists that contain dicts."""
        config = {"items": [{"model": "$model/a"}, {"model": "$model/b"}]}
        expand_subfolders(config, model_id="org/repo", parent_subfolder="p")
        assert config["items"][0]["model"]["subfolder"] == "p/a"
        assert config["items"][1]["model"]["subfolder"] == "p/b"

    def test_no_model_id_leaves_checkpoint_as_none(self):
        """When model_id is None the checkpoint field is also None."""
        config = {"seg": "$model/seg"}
        expand_subfolders(config, model_id=None)
        assert config["seg"]["checkpoint"] is None

    def test_parent_subfolder_combined_with_explicit_revision(self):
        """parent_subfolder and explicit @revision can coexist."""
        config = {"model": "$model/weights@pinned"}
        expand_subfolders(
            config,
            model_id="org/repo",
            parent_subfolder="folder",
            parent_revision="ignored",
        )
        assert config["model"]["subfolder"] == "folder/weights"
        assert config["model"]["revision"] == "pinned"


# ---------------------------------------------------------------------------
# Pipeline.from_pretrained local-directory tests
# ---------------------------------------------------------------------------


class TestFromPretrainedLocalSubfolder:
    """Pipeline.from_pretrained with local directories and the subfolder arg."""

    def _minimal_config(self) -> dict:
        return {"pipeline": {"name": _DUMMY_CLS_DOTPATH}}

    def test_loads_config_from_root_without_subfolder(self, tmp_path):
        """Without subfolder, config.yaml is read from the directory root."""
        write_config(tmp_path, self._minimal_config())
        pipeline = Pipeline.from_pretrained(str(tmp_path))
        assert isinstance(pipeline, _DummyPipeline)

    def test_loads_config_from_subfolder(self, tmp_path):
        """With subfolder='sub', config.yaml is read from <root>/sub/config.yaml."""
        write_config(tmp_path, self._minimal_config(), subfolder="sub")
        pipeline = Pipeline.from_pretrained(str(tmp_path), subfolder="sub")
        assert isinstance(pipeline, _DummyPipeline)

    def test_subfolder_does_not_exist_raises(self, tmp_path):
        """Requesting a non-existent subfolder should raise (file not found)."""
        write_config(tmp_path, self._minimal_config())
        with pytest.raises(Exception):
            Pipeline.from_pretrained(str(tmp_path), subfolder="missing")

    def test_model_refs_resolved_under_subfolder(self, tmp_path):
        """$model/seg refs in config are expanded to <subfolder>/seg."""
        config = {
            "pipeline": {"name": _DUMMY_CLS_DOTPATH},
            "params": {"segmentation": "$model/seg"},
        }
        write_config(tmp_path, config, subfolder="v1")

        # Test expand_subfolders directly since loading would require real weights.
        expand_subfolders(config, model_id=str(tmp_path), parent_subfolder="v1")
        assert config["params"]["segmentation"]["subfolder"] == "v1/seg"
        assert str(config["params"]["segmentation"]["checkpoint"]) == str(tmp_path)

    def test_nested_subfolder_path(self, tmp_path):
        """subfolder can be a multi-level path like 'a/b'."""
        write_config(tmp_path, self._minimal_config(), subfolder="a/b")
        pipeline = Pipeline.from_pretrained(str(tmp_path), subfolder="a/b")
        assert isinstance(pipeline, _DummyPipeline)

    def test_model_refs_resolved_under_nested_subfolder(self, tmp_path):
        """$model/x under nested subfolder 'a/b' resolves to 'a/b/x'."""
        config = {
            "pipeline": {"name": _DUMMY_CLS_DOTPATH},
            "params": {"embedding": "$model/emb"},
        }
        expand_subfolders(config, model_id="org/repo", parent_subfolder="a/b")
        assert config["params"]["embedding"]["subfolder"] == "a/b/emb"

    def test_revision_raises_with_local_directory(self, tmp_path):
        """Passing revision= with a local directory must raise ValueError."""
        write_config(tmp_path, self._minimal_config())
        with pytest.raises(ValueError, match="[Rr]evision"):
            Pipeline.from_pretrained(str(tmp_path), revision="main")


# ---------------------------------------------------------------------------
# Nested pipeline tests: parent → sub → sub
# ---------------------------------------------------------------------------


class TestNestedSubfolderPipelines:
    """Parent pipeline loads a subpipeline which itself loads a subpipeline.

    Directory layout used in these tests:

        <root>/
          v1/
            config.yaml          ← _DummyParentPipeline, sub: "$model/child"
            child/
              config.yaml        ← _DummyParentPipeline, sub: "$model/grandchild"
              grandchild/
                config.yaml      ← _DummyPipeline (leaf)

    Loading with Pipeline.from_pretrained(<root>, subfolder="v1") must:
      - expand "$model/child"      → subfolder "v1/child"
      - expand "$model/grandchild" → subfolder "v1/child/grandchild"
    """

    def _write_three_level_tree(self, root: Path) -> None:
        write_config(
            root,
            {"pipeline": {"name": _DUMMY_PARENT_CLS_DOTPATH, "params": {"sub": "$model/child"}}},
            subfolder="v1",
        )
        write_config(
            root,
            {"pipeline": {"name": _DUMMY_PARENT_CLS_DOTPATH, "params": {"sub": "$model/grandchild"}}},
            subfolder="v1/child",
        )
        write_config(
            root,
            {"pipeline": {"name": _DUMMY_CLS_DOTPATH}},
            subfolder="v1/child/grandchild",
        )

    def test_parent_is_correct_type(self, tmp_path):
        """Top-level pipeline is a _DummyParentPipeline."""
        self._write_three_level_tree(tmp_path)
        parent = Pipeline.from_pretrained(str(tmp_path), subfolder="v1")
        assert isinstance(parent, _DummyParentPipeline)

    def test_subpipeline_is_registered(self, tmp_path):
        """Sub-pipeline is stored in parent._pipelines under the 'sub' key."""
        self._write_three_level_tree(tmp_path)
        parent = Pipeline.from_pretrained(str(tmp_path), subfolder="v1")
        assert "sub" in parent._pipelines

    def test_subpipeline_is_correct_type(self, tmp_path):
        """Sub-pipeline is itself a _DummyParentPipeline."""
        self._write_three_level_tree(tmp_path)
        parent = Pipeline.from_pretrained(str(tmp_path), subfolder="v1")
        assert isinstance(parent.sub, _DummyParentPipeline)

    def test_grandchild_pipeline_is_registered(self, tmp_path):
        """Grand-child pipeline is stored in child._pipelines under 'sub'."""
        self._write_three_level_tree(tmp_path)
        parent = Pipeline.from_pretrained(str(tmp_path), subfolder="v1")
        assert "sub" in parent.sub._pipelines

    def test_grandchild_pipeline_is_leaf_type(self, tmp_path):
        """Grand-child pipeline is a _DummyPipeline (the leaf class)."""
        self._write_three_level_tree(tmp_path)
        parent = Pipeline.from_pretrained(str(tmp_path), subfolder="v1")
        assert isinstance(parent.sub.sub, _DummyPipeline)

    def test_without_subfolder_loads_from_root(self, tmp_path):
        """Without subfolder=, the root config.yaml is used (one-level tree)."""
        write_config(
            tmp_path,
            {"pipeline": {"name": _DUMMY_PARENT_CLS_DOTPATH, "params": {"sub": "$model/child"}}},
        )
        write_config(tmp_path, {"pipeline": {"name": _DUMMY_CLS_DOTPATH}}, subfolder="child")
        parent = Pipeline.from_pretrained(str(tmp_path))
        assert isinstance(parent, _DummyParentPipeline)
        assert isinstance(parent.sub, _DummyPipeline)
