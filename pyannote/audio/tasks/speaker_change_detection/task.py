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
import scipy.signal

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class SpeakerChangeDetection(Task):
    """Speaker change detection

    Speaker change detection is the task of detecting speaker change points
    in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except "speech" class is replaced by "change" where a frame is marked as
    "change" if a speaker change happens less than `collar` frames away.

    Note that non-speech/speech changes are not marked as speaker change.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    collar : int, optional.
        Mark frames less than `collar` frames away from actual change point as positive.
        Defaults to 1.
    batch_size : int, optional
        Number of training samples per batch.
    num_workers : int, optional
        Number of workers used for generating training samples.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        collar: int = 1,
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
            problem=Problem.BINARY_CLASSIFICATION,
            scale=Scale.FRAME,
            duration=self.duration,
            classes=[
                "change",
            ],
        )

        self.collar = collar

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

    def prepare_y(self, one_hot_y: np.ndarray, collar: int = None):
        """Get speaker change detection targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.
        collar : int, optional
            Mark frames less than `collar` frames away from actual change point as positive.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is a change of speaker at tth frame, 0 otherwise.
        """

        if collar is None:
            collar = self.collar

        num_frames, num_speakers = one_hot_y.shape

        #  y[t] = True if speaker change, False otherwise
        y = np.sum(np.abs(np.diff(one_hot_y, axis=0)), axis=1, keepdims=True)
        y = np.vstack(([[0]], y > 0))

        # mark frames in the neighborhood of actual change point as positive.
        window = scipy.signal.triang(2 * collar + 1)[:, np.newaxis]
        y = np.minimum(1, scipy.signal.convolve(y, window, mode="same"))
        y = 1 * (y > 1e-10)

        # at this point, all segment boundaries are marked as change, including non-speech/speaker changes.
        # let's remove non-speech/speaker change

        # append empty samples at the beginning/end
        expanded_y = np.vstack(
            [
                np.zeros((collar, num_speakers), dtype=one_hot_y.dtype),
                one_hot_y,
                np.zeros((collar, num_speakers), dtype=one_hot_y.dtype),
            ]
        )

        # stride trick. data[i] is now a sliding window of collar length
        # centered at time step i.
        data = np.lib.stride_tricks.as_strided(
            expanded_y,
            shape=(num_frames, num_speakers, 2 * collar + 1),
            strides=(one_hot_y.strides[0], one_hot_y.strides[1], one_hot_y.strides[0]),
        )

        # y[i] = 1 if more than one speaker are speaking in the
        # corresponding window. 0 otherwise
        x_speakers = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
        x_speakers = x_speakers.reshape(-1, 1)

        y *= x_speakers

        return np.squeeze(y)

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

            y = self.prepare_y(one_hot_y, collar=self.collar)

            yield {"X": X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)
