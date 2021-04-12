from glob import glob

import papermill as pm


def test_can_run_notebooks():
    # Search for all notebooks in directory
    notebooks = glob("**/notebook/**/*.ipynb")
    for nb in notebooks:
        try:
            pm.execute_notebook(
                nb, "/dev/null", progress_bar=False, kernel_name="python"
            )
        except Exception as e:
            # Which notebook caused the error
            raise Exception(nb, e)
