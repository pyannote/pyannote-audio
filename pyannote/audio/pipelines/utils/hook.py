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

from copy import deepcopy
from typing import Any, Mapping, Optional, Text

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def logging_hook(
    step_name: Text,
    step_artifact: Any,
    file: Optional[Mapping] = None,
    completed: Optional[int] = None,
    total: Optional[int] = None,
):
    """Hook to save step_artifact as file[step_name]

    Useful for debugging purposes
    """

    if completed is None:
        file[step_name] = deepcopy(step_artifact)


class ProgressHook:
    """Hook to show progress of each internal step

    Example
    -------
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    with ProgressHook() as hook:
       output = pipeline(file, hook=hook)
    """

    def __enter__(self):

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
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
