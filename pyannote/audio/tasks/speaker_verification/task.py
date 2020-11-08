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
from itertools import chain
from typing import TYPE_CHECKING

import pytorch_metric_learning.losses
import torch.optim

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol

if TYPE_CHECKING:
    from pyannote.audio.core.model import Model


class SpeakerEmbeddingArcFace(Task):
    """


    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.


    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        batch_size: int = None,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # there is no such thing as a "class" in representation
        # learning, so we do not need to define it here.
        self.specifications = TaskSpecification(
            problem=Problem.REPRESENTATION,
            scale=Scale.CHUNK,
            duration=self.duration,
        )

    def setup(self, stage=None):

        if stage == "fit":

            # gather training set metadata
            self.speakers = dict()
            for f in self.protocol.train():

                for speaker in f["annotation"].labels():

                    # keep speaker's (long enough) speech turns...
                    speech_turns = [
                        segment
                        for segment in f["annotation"].label_timeline(speaker)
                        if segment.duration > self.duration
                    ]

                    # skip if there is no speech turns left
                    if not speech_turns:
                        continue

                    # ... and their total duration
                    duration = sum(segment.duration for segment in speech_turns)

                    # add speaker to the list of speakers
                    if speaker not in self.speakers:
                        self.speakers[speaker] = list()

                    self.speakers[speaker].append(
                        {
                            "audio": f["audio"],
                            "duration": duration,
                            "speech_turns": speech_turns,
                        }
                    )

            # for convenience, we keep track of the list of speakers, after all
            self.specifications.classes = sorted(self.speakers)

            num_classes = len(self.speakers)
            # use example_output_array to guess embedding size
            _, embedding_size = self.example_output_array.shape
            self.loss_func = pytorch_metric_learning.losses.ArcFaceLoss(
                num_classes, embedding_size, margin=28.6, scale=64
            )

    def train__iter__(self):
        """Iterate over training samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: int
            Speaker index.
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker()

        speakers = list(self.speakers)

        while True:

            # shuffle speakers so that we don't always have the same
            # groups of speakers in a batch (which might be especially
            # problematic for contrast-based losses like contrastive
            # or triplet loss.
            rng.shuffle(speakers)

            for speaker in speakers:

                # speaker index in original sorted order
                y = self.specifications.classes.index(speaker)

                # three chunks per speaker
                for _ in range(3):

                    # select one file at random (with probability proportional to its speaker duration)
                    file, *_ = rng.choices(
                        self.speakers[speaker],
                        weights=[f["duration"] for f in self.speakers[speaker]],
                        k=1,
                    )

                    # select one speech turn at random (with probability proportional to its duration)
                    speech_turn, *_ = rng.choices(
                        file["speech_turns"],
                        weights=[s.duration for s in file["speech_turns"]],
                        k=1,
                    )

                    # select one chunk at random (with uniform distribution)
                    start_time = rng.uniform(
                        speech_turn.start, speech_turn.end - self.duration
                    )
                    chunk = Segment(start_time, start_time + self.duration)

                    X = self.prepare_chunk(file, chunk, duration=self.duration)

                    yield {"X": X, "y": y}

    def train__len__(self):
        duration = sum(
            datum["duration"] for data in self.speakers.values() for datum in data
        )
        return math.ceil(duration / self.duration)

    def training_step(self, model: "Model", batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        loss = self.loss_func(model(X), y)
        model.log("train_loss", loss)
        return loss

    def configure_optimizers(self, model: "Model"):
        parameters = chain(model.parameters(), self.loss_func.parameters())
        return torch.optim.Adam(parameters, lr=1e-3)
