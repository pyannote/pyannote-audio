# Development

This uv workspace contains five independently installable packages under `packages/`.
Python 3.11 or newer is required. Audio additionally needs a compatible FFmpeg installation.

## Work on the workspace

From the repository root:

```bash
uv sync --all-packages --extra test --extra cli
uv run --package pyannote-audio python -c "from pyannote.audio import Pipeline"
```

Workspace sources resolve dependencies to the local packages rather than PyPI.
`uv.lock` is generated locally and ignored. To use optional features, select their
extras, for example `uv sync --all-packages --extra test --extra cli --extra transcription`.
The packages retain their own dependency and extra definitions.

Tests use package-relative fixtures and some share module names, so run suites in
separate processes from their package directories:

```bash
cd packages/database
uv run --package pyannote-database --extra test --extra cli --extra transcription pytest tests
```

For audio, exclude model downloads and historical notebooks during ordinary tests:

```bash
cd packages/audio
uv run --package pyannote-audio --extra test --extra cli pytest tests -m "not network and not notebook"
```

Use `-m network` to test the public Hugging Face model. Historical notebooks remain
opt-in with `-m notebook`. The separation extra has a separate CI smoke test.

## Test the distributions together

From the repository root:

```bash
uv build --all-packages
uv run --no-project --with packaging python -m scripts.release
uv venv .wheel-venv --python 3.11
uv run --no-project --python .wheel-venv/bin/python python -m scripts.install_wheels --scope all
uv run --no-project --python .wheel-venv/bin/python python -m scripts.test_packages --scope all
```

On Linux, add `--cpu` to the installation helper to use the CPU PyTorch wheels.
The helper installs all sibling wheels explicitly. `--scope libraries` omits audio;
`optional`, `network`, and `separation` select the corresponding CI checks.

Run `uvx ruff==0.17.1 check .` and `uvx ruff==0.17.1 format --check .` before submitting.
Package-specific Ruff settings remain in effect. Packaging/release guard tests run with:

```bash
uv run --no-project --with pytest --with packaging python -m pytest tests/monorepo
```

Record future user-visible changes in the root `CHANGELOG.md`, identifying the affected
package. Package changelogs preserve pre-monorepo release history.
