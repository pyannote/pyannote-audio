# Copyright (c) 2025- pyannoteAI
"""Build all five documentation sites under a shared landing page."""

import subprocess
import sys
from pathlib import Path

from scripts._common import PACKAGES

root = Path(__file__).resolve().parents[1]
site = root / "site"
site.mkdir(exist_ok=True)
for name in PACKAGES:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "doc/source",
            str(site / name),
        ],
        cwd=root / "packages" / name,
        check=True,
    )
links = "".join(f'<li><a href="{name}/">pyannote.{name}</a></li>' for name in PACKAGES)
(site / "index.html").write_text(
    f'<!doctype html><html lang="en"><meta charset="utf-8"><title>pyannote documentation</title><h1>pyannote documentation</h1><ul>{links}</ul><p><a href="https://github.com/pyannote/pyannote-audio">Source and releases</a></p></html>'
)
(site / ".nojekyll").touch()
