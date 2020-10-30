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

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class SpeakerTracking(Task):
    """Speaker tracking

    Speaker tracking is the process of determining if and when a (previously
    enrolled) person's voice can be heard in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"speaker1", ...,
    "speaker_N"} where N is the number of speakers in the training set.

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

        # for speaker tracking, task specification depends
        # on the data: we do not know in advance which
        # speakers should be tracked. therefore, we postpone
        # the definition of specifications.

    def setup(self, stage=None):

        if stage == "fit":

            # loop over the training set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.

            # also build the list of speakers to be tracked.

            self.train, speakers = [], set()
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
                speakers.update(f["annotation"].labels())

        # now that we now who the speakers are, we can
        # define the task specifications.

        # note that, since multiple speakers can be active
        # at once, the problem is multi-label classification.
        self.specifications = TaskSpecification(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            scale=Scale.FRAME,
            duration=self.duration,
            classes=sorted(speakers),
        )

    def train__iter__(self):
        """Iterate over training samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: (frame, num_speakers)
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
                labels=self.specifications.classes,
            )

            yield {"X": X, "y": one_hot_y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)
