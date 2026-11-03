# Copyright (c) 2025- pyannoteAI
"""Install workspace wheels together so tests never fall back to published siblings."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts._common import PACKAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument(
        "--scope",
        choices=["all", "libraries", "optional", "network", "separation", "docs"],
        default="all",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    names = PACKAGES[:-1] if args.scope == "libraries" else PACKAGES
    wheels = []
    for name in names:
        matches = list(args.dist.glob(f"pyannote_{name}-*.whl"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} wheel, found {matches}")
        extras = ["test"]
        if args.scope == "docs":
            extras = ["doc"]
            if name != "core":
                extras += ["cli"]
        elif args.scope == "optional":
            extras += {
                "core": ["notebook"],
                "database": ["cli", "transcription"],
                "metrics": ["cli", "plot", "transcription"],
                "pipeline": ["cli"],
                "audio": ["cli"],
            }[name]
        elif name == "audio":
            extras += ["cli"]
            if args.scope == "separation":
                extras += ["separation"]
        wheels.append(f"{matches[0].resolve()}[{','.join(extras)}]")
    command = ["uv", "pip", "install", "--python", sys.executable]
    if args.cpu:
        command += ["--torch-backend", "cpu"]
    subprocess.run([*command, *wheels], check=True)


if __name__ == "__main__":
    main()
