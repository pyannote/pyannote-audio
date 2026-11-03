"""Opt-in checks for historical notebooks: pytest -m notebook."""

from pathlib import Path

import papermill as pm
import pytest


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", sorted(Path("notebook").glob("*.ipynb")))
def test_can_run_notebook(notebook, tmp_path):
    pm.execute_notebook(
        notebook, tmp_path / notebook.name, progress_bar=False, kernel_name="python"
    )
