# MIT License
#
# Copyright (c) 2020 CNRS
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

import math

import numpy as np

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class VoiceActivityDetection(Task):
    """Voice activity detection

    Voice activity detection (or VAD) is the task of detecting speech regions
    in a given audio recording.

    It is addressed as a binary {"non-speech", "speech"} sequence labeling task.
    A frame is marked as "speech" as soon as at least one speaker is active.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    batch_size : int, optional
        Number of training samples per batch.
    num_workers : int, optional
        Number of workers used for generating training samples.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        batch_size: int = None,
        num_workers: int = 1,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.specifications = TaskSpecification(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            scale=Scale.FRAME,
            duration=self.duration,
            classes=["non_speech", "speech"],
        )

    def setup(self, stage=None):
        if stage == "fit":
            # loop over the training set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.
            self.train = []
            for f in self.protocol.train():
                segments = [
                    segment
                    for segment in f["annotated"]
                    if segment.duration > self.duration
                ]
                duration = sum(segment.duration for segment in segments)
                self.train.append(
                    {
                        "annotated": segments,
                        "annotation": f["annotation"],
                        "duration": duration,
                        "audio": f["audio"],
                    }
                )

    def prepare_y(self, one_hot_y: np.ndarray):
        """Get voice activity detection targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if at least one speaker is active at tth frame, 0 otherwise.
        """
        return np.int64(np.sum(one_hot_y, axis=1) > 0)

    def train__iter__(self):
        """Iterate over training samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: (frame, )
            Frame-level targets. Note that frame < time.
            `frame` is infered automagically from the
            example model output.
        """
        # create worker-specific random number generator
        rng = create_rng_for_worker()

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                self.train,
                weights=[f["duration"] for f in self.train],
                k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = rng.choices(
                file["annotated"],
                weights=[s.duration for s in file["annotated"]],
                k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = rng.uniform(segment.start, segment.end - self.duration)
            chunk = Segment(start_time, start_time + self.duration)

            X, one_hot_y, _ = self.prepare_chunk(
                file,
                chunk,
                duration=self.duration,
                return_y=True,
            )

            y = self.prepare_y(one_hot_y)

            yield {"X": X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)
