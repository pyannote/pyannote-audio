# The MIT License (MIT)

# Copyright (c) 2020- CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Alexis PLAQUET


import math
from enum import StrEnum

import typer
from pyannote.core import Annotation

from pyannote.database import Database, registry
from pyannote.database.protocol import CollectionProtocol, SpeakerDiarizationProtocol

app = typer.Typer()


class Task(StrEnum):
    Any = "Any"
    Protocol = "Protocol"
    Collection = "Collection"
    SpeakerDiarization = "SpeakerDiarization"
    SpeakerVerification = "SpeakerVerification"


@app.command("database")
def database():
    """Print list of databases"""
    for database in registry.databases:
        typer.echo(f"{database}")


@app.command("task")
def task(
    database: str = typer.Option(
        "",
        "--database",
        "-d",
        metavar="DATABASE",
        help="Filter tasks by DATABASE.",
        case_sensitive=False,
    ),
):
    """Print list of tasks"""

    if database == "":
        tasks = []
    else:
        db: Database = registry.get_database(database)
        tasks = db.get_tasks()

    for task in tasks:
        typer.echo(f"{task}")


@app.command("protocol")
def protocol(
    database: str = typer.Option(
        "",
        "--database",
        "-d",
        metavar="DATABASE",
        help="Filter protocols by DATABASE.",
        case_sensitive=False,
    ),
    task: Task = typer.Option(
        "Any",
        "--task",
        "-t",
        help="Filter protocols by TASK.",
        case_sensitive=False,
    ),
):
    """Print list of protocols"""

    if database == "":
        databases = list(registry.databases)
    else:
        databases = [database]

    for database_name in databases:
        db: Database = registry.get_database(database_name)
        tasks = db.get_tasks() if task == "Any" else [task]
        for task_name in tasks:
            try:
                protocols = db.get_protocols(task_name)
            except KeyError:
                continue
            for protocol in protocols:
                typer.echo(f"{database_name}.{task_name}.{protocol}")


def duration_to_str(seconds: float) -> str:
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds - 3600 * hours) / 60)
    return f"{hours}h{minutes:02d}m"


@app.command("info")
def info(protocol: str):
    """Print protocol detailed information"""

    p = registry.get_protocol(protocol)

    if isinstance(p, SpeakerDiarizationProtocol):
        subsets = ["train", "development", "test"]
        skip_annotation = False
        skip_annotated = False
    elif isinstance(p, CollectionProtocol):
        subsets = ["files"]
        skip_annotation = True
        skip_annotated = True
    else:
        typer.echo("Only collections and speaker diarization protocols are supported.")
        typer.Exit(code=1)

    for subset in subsets:
        num_files = 0
        speakers = set()
        duration = 0.0
        speech = 0.0

        def iterate():
            try:
                yield from getattr(p, subset)()
            except (AttributeError, NotImplementedError):
                return

        for file in iterate():
            num_files += 1

            if not skip_annotation:
                annotation = file["annotation"] or Annotation(uri=file["uri"])
                speakers.update(annotation.labels())
                speech += annotation.get_timeline().support().duration()

            if not skip_annotated:
                annotated = file["annotated"]
                duration += annotated.duration()

        if num_files > 0:
            typer.secho(
                f"{subset}", fg=typer.colors.BRIGHT_GREEN, underline=True, bold=True
            )
            typer.echo(f"   {num_files} files")
            if not skip_annotated:
                typer.echo(f"   {duration_to_str(duration)} annotated")

            if not skip_annotation:
                typer.echo(
                    f"   {duration_to_str(speech)} of speech ({100 * speech / duration:.0f}%)"
                )
                typer.echo(f"   {len(speakers)} speakers")


def main():
    app()
