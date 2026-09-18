"""matplotlib is an optional dependency (``pyannote.audio[plot]``).

It is only used to draw validation samples during training and to render
``utils.preview``, so no module of ``pyannote.audio`` may import it at module
level: an inference-only install has no matplotlib, and where it is installed,
importing ``matplotlib.pyplot`` costs a font-cache build on first import
(tens of seconds on a cold cache, and a hard crash on macOS with
matplotlib < 3.12, see matplotlib#32328) that ``Pipeline`` users never asked
for.  Imports belong inside the functions that plot.
"""

import ast
from pathlib import Path

import pyannote.audio

PACKAGE = Path(pyannote.audio.__file__).parent


def _module_level_matplotlib_imports(source: str) -> list[int]:
    tree = ast.parse(source)
    lines = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] == "matplotlib" for name in names):
            lines.append(node.lineno)
    return lines


def test_no_module_level_matplotlib_import():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for lineno in _module_level_matplotlib_imports(path.read_text()):
            offenders.append(f"{path.relative_to(PACKAGE)}:{lineno}")
    assert not offenders, (
        "matplotlib must be imported inside the functions that plot, "
        "not at module level:\n  " + "\n  ".join(offenders)
    )
