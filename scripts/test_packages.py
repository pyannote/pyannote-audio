# Copyright (c) 2025- pyannoteAI
"""Run each package's tests separately, preserving fixture paths and module names."""

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from scripts._common import PACKAGES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["all", "libraries", "optional", "network", "separation"],
        default="all",
    )
    args = parser.parse_args()
    names = PACKAGES[:-1] if args.scope == "libraries" else PACKAGES
    versions = {importlib.metadata.version(f"pyannote-{name}") for name in names}
    if len(versions) != 1:
        raise ValueError(
            f"Tests require a synchronized set of installed packages: {versions}"
        )
    # Test artifacts, not editable checkouts. Verify this without importing audio/FFmpeg.
    for name in names:
        distribution = importlib.metadata.distribution(f"pyannote-{name}")
        if not Path(distribution.locate_file(f"pyannote/{name}/__init__.py")).is_file():
            raise ValueError(f"{name} is not installed from a wheel")
    for name in names:
        tests = ["tests"]
        if args.scope == "network":
            if name != "audio":
                continue
            tests += ["-m", "network"]
        elif args.scope == "separation":
            if name != "audio":
                continue
            tests = ["tests/test_separation.py", "tests/test_totatonet.py"]
        elif args.scope == "optional":
            if name in {"database", "metrics"}:
                tests = ["tests/test_optional_features.py"]
            elif name == "audio":
                tests = ["tests/test_cli.py"]
        if name == "audio" and args.scope == "all":
            tests += ["-m", "not network and not notebook"]
        subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-q", "--tb=short"],
            cwd=ROOT / "packages" / name,
            check=True,
        )
    if args.scope == "optional":
        subprocess.run(
            [str(Path(sys.executable).with_name("pyannote-pipeline")), "--help"],
            check=True,
        )


if __name__ == "__main__":
    main()
