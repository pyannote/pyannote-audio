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
import warnings
from typing import List, Optional, Text, Tuple

import numpy as np

from pyannote.audio.core.io import AudioFile
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment, SlidingWindow


class SegmentationTaskMixin:
    """Methods common to most segmentation tasks"""

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
                        "uri": f["uri"],
                        "annotated": segments,
                        "annotation": f["annotation"],
                        "duration": duration,
                        "audio": f["audio"],
                    }
                )

            # loop over the validation set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.
            self.validation = []
            for f in self.protocol.development():
                segments = [
                    segment
                    for segment in f["annotated"]
                    if segment.duration > self.duration
                ]
                num_chunks = sum(
                    round(segment.duration // self.duration) for segment in segments
                )
                self.validation.append(
                    {
                        "uri": f["uri"],
                        "annotated": segments,
                        "annotation": f["annotation"],
                        "num_chunks": num_chunks,
                        "audio": f["audio"],
                    }
                )

    def prepare_y(self, one_hot_y: np.ndarray) -> np.ndarray:
        return one_hot_y

    @property
    def chunk_labels(self) -> Optional[List[Text]]:
        """Ordered list of labels

        Override this method to make `prepare_chunk` use a specific
        ordered list of labels when extracting frame-wise labels.

        See `prepare_chunk` source code for details.
        """
        return None

    def prepare_chunk(
        self,
        file: AudioFile,
        chunk: Segment,
        duration: float = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Text]]:
        """Extract audio chunk and corresponding frame-wise labels

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment
            Audio chunk.
        duration : float, optional
            Fix chunk duration to avoid rounding errors. Defaults to self.duration

        Returns
        -------
        X : np.ndarray
            Audio chunk as (num_samples, num_channels) array.
        y : np.ndarray, optional
            Frame-wise labels as (num_frames, num_labels) array.
        labels : list of str, optional
            Ordered labels such that y[:, k] corresponds to activity of labels[k].
        """

        X, _ = self.audio.crop(
            file,
            chunk,
            mode="center",
            fixed=self.duration if duration is None else duration,
        )

        if self.is_multi_task:
            # this assumes that all tasks share the same model introspection.
            # this is a reasonable assumption for now.
            any_task = next(iter(self.model_introspection.keys()))
            num_frames, _ = self.model_introspection[any_task](X.shape[1])
        else:
            num_frames, _ = self.model_introspection(X.shape[1])

        annotation = file["annotation"].crop(chunk)
        labels = annotation.labels() if self.chunk_labels is None else self.chunk_labels

        y = np.zeros((num_frames, len(labels)), dtype=np.int8)
        frames = SlidingWindow(
            start=chunk.start,
            duration=self.duration / num_frames,
            step=self.duration / num_frames,
        )
        for label in annotation.labels():
            try:
                k = labels.index(label)
            except ValueError:
                warnings.warn(
                    f"File {file['uri']} contains unexpected label '{label}'."
                )
                continue

            segments = annotation.label_timeline(label)
            for start, stop in frames.crop(segments, mode="center", return_ranges=True):
                y[start:stop, k] += 1

        # handle corner case when the same label is active more than once
        y = np.minimum(y, 1, out=y)

        return X, y, labels

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

            X, one_hot_y, _ = self.prepare_chunk(file, chunk, duration=self.duration)

            y = self.prepare_y(one_hot_y)

            yield {"X": X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)

    def val__iter__(self):
        """Iterate over validation samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: (frame, )
            Frame-level targets. Note that frame < time.
            `frame` is infered automagically from the
            example model output.
        """

        for f in self.validation:

            for segment in f["annotated"]:

                for c in range(f["num_chunks"]):
                    start_time = segment.start + c * self.duration
                    chunk = Segment(start_time, start_time + self.duration)

                    X, one_hot_y, _ = self.prepare_chunk(
                        f, chunk, duration=self.duration
                    )

                    y = self.prepare_y(one_hot_y)

                    yield {"X": X, "y": y}

    def val__len__(self):
        # Number of validation samples in one epoch
        num_chunks = sum(file["num_chunks"] for file in self.validation)
        return num_chunks
