# MIT License
#
# Copyright (c) 2022- CNRS
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

import time
from copy import deepcopy
from typing import Any, Mapping, Optional, Text

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class ArtifactHook:
    """Hook to save artifacts of each internal step

    Parameters
    ----------
    artifacts: list of str, optional
        List of steps to save. Defaults to all steps.
    file_key: str, optional
        Key used to store artifacts in `file`.
        Defaults to "artifact".

    Usage
    -----
    >>> with ArtifactHook() as hook:
    ...     output = pipeline(file, hook=hook)
    # file["artifact"] contains a dict with artifacts of each step

    """

    def __init__(self, *artifacts, file_key: str = "artifact"):
        self.artifacts = artifacts
        self.file_key = file_key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        if (step_artifact is None) or (
            self.artifacts and step_name not in self.artifacts
        ):
            return

        file.setdefault(self.file_key, dict())[step_name] = deepcopy(step_artifact)


class ProgressHook:
    """Hook to show progress of each internal step

    Parameters
    ----------
    transient: bool, optional
        Clear the progress on exit. Defaults to False.

    Example
    -------
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    with ProgressHook() as hook:
       output = pipeline(file, hook=hook)
    """

    def __init__(self, transient: bool = False):
        self.transient = transient

    def __enter__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            transient=self.transient,
        )
        self.progress.start()
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        if completed is None:
            completed = total = 1

        if not hasattr(self, "step_name") or step_name != self.step_name:
            self.step_name = step_name
            self.step = self.progress.add_task(self.step_name)

        self.progress.update(self.step, completed=completed, total=total)

        # force refresh when completed
        if completed >= total:
            self.progress.refresh()


class TimingHook:
    """Hook to compute processing time of internal steps

    Parameters
    ----------
    file_key: str, optional
        Key used to store processing time in `file`.
        Defaults to "timing_hook".

    Usage
    -----
    >>> with TimingHook() as hook:
    ...     output = pipeline(file, hook=hook)
    # file["timing_hook"]  contains processing time for each step
    """

    def __init__(self, file_key: str = "timing"):
        self.file_key = file_key

    def __enter__(self):
        self._pipeline_start_time = time.time()
        self._start_time = dict()
        self._end_time = dict()
        return self

    def __exit__(self, *args):
        _pipeline_end_time = time.time()
        processing_time = dict()
        processing_time["total"] = _pipeline_end_time - self._pipeline_start_time
        for step_name, _start_time in self._start_time.items():
            _end_time = self._end_time[step_name]
            processing_time[step_name] = _end_time - _start_time

        self._file[self.file_key] = processing_time

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        if not hasattr(self, "_file"):
            self._file = file

        if completed is None:
            return

        if completed == 0:
            self._start_time[step_name] = time.time()

        if completed >= total:
            self._end_time[step_name] = time.time()


class Hooks:
    """List of hooks

    Usage
    -----
    >>> with Hooks(ProgessHook(), TimingHook(), ArtifactHook()) as hook:
    ...     output = pipeline("audio.wav", hook=hook)

    """

    def __init__(self, *hooks):
        self.hooks = hooks

    def __enter__(self):
        for hook in self.hooks:
            if hasattr(hook, "__enter__"):
                hook.__enter__()
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            if hasattr(hook, "__exit__"):
                hook.__exit__(*args)

    def __call__(
        self,
        step_name: Text,
        step_artifact: Any,
        file: Optional[Mapping] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        for hook in self.hooks:
            hook(step_name, step_artifact, file=file, total=total, completed=completed)
