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


import random
from typing import Mapping

import numpy as np

from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Task
from pyannote.audio.tasks import (
    OverlappedSpeechDetection,
    SpeakerChangeDetection,
    VoiceActivityDetection,
)
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class MultiTaskSegmentation(SegmentationTaskMixin, Task):
    """Multi-task segmentation

    Multi-task training of segmentation tasks, including:
        - vad: voice activity detection
        - scd: speaker change detection
        - osd: overlapped speech detection

    Note that, when "osd" is one of the tasks considered, data augmentation is used
    to artificially increase the proportion of "overlap". This is achieved by
    generating chunks made out of the (weighted) sum of two random chunks.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    vad : bool, optional
        Add voice activity detection in the pool of tasks.
        Defaults to False.
    vad_params : dict, optional
        Additional vad-specific parameters. Has no effect when `vad` is False.
        See VoiceActivityDetection docstring for details and default value.
    scd : bool, optional
        Add speaker change detection to the pool of tasks.
        Defaults to False.
    scd_params : dict, optional
        Additional scd-specific parameters. Has no effect when `scd` is False.
        See SpeakerChangeDetection docstring for details and default value.
    osd : bool, optional
        Add overlapped speech detection to the pool of tasks.
        Defaults to False.
    osd_params : dict, optional
        Additional osd-specific parameters. Has no effect when `osd` is False.
        See OverlappedSpeechDetection docstring for details and default value.
    batch_size : int, optional
        Number of training samples per batch.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        vad: bool = False,
        vad_params: Mapping = None,
        scd: bool = False,
        scd_params: Mapping = None,
        osd: bool = False,
        osd_params: Mapping = None,
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

        self.vad = vad
        self.vad_params = dict() if vad_params is None else vad_params

        self.scd = scd
        self.scd_params = dict() if scd_params is None else scd_params

        self.osd = osd
        self.osd_params = dict() if osd_params is None else osd_params

        if sum([vad, scd, osd]) < 2:
            raise ValueError(
                "You must activate at least two tasks among 'vad', 'scd', and 'osd'."
            )

        self.tasks = dict()
        if self.vad:
            self.tasks["vad"] = VoiceActivityDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.vad_params,
            )
        if self.scd:
            self.tasks["scd"] = SpeakerChangeDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.scd_params,
            )
        if self.osd:
            self.tasks["osd"] = OverlappedSpeechDetection(
                protocol,
                duration=duration,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **self.osd_params,
            )

    def setup(self, stage=None):

        super().setup(stage=stage)

        if stage == "fit":

            if self.osd and self.tasks["osd"].domain is not None:
                for f in self.train:
                    f["domain"] = f[self.tasks["osd"].domain]
                self.domains = list(set(f["domain"] for f in self.train))

            self.specifications = {
                name: task.specifications for name, task in self.tasks.items()
            }

    def prepare_y(self, one_hot_y: np.ndarray):
        """Get multi-task targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is two or more active speakers at tth frame, 0 otherwise.
        """

        return {name: task.prepare_y(one_hot_y) for name, task in self.tasks.items()}

    def train__iter__helper(self, rng: random.Random, domain: str = None):

        train = self.train

        if domain is not None:
            train = [f for f in train if f["domain"] == domain]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                train,
                weights=[f["duration"] for f in train],
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

            yield self.prepare_chunk(file, chunk, duration=self.duration)

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

        if self.osd and self.tasks["osd"].domain is not None:
            chunks_by_domain = {
                domain: self.train__iter__helper(rng, domain=domain)
                for domain in self.domains
            }
        else:
            chunks = self.train__iter__helper(rng)

        while True:

            if self.osd and self.tasks["osd"].domain is not None:
                #  draw domain at random
                domain = rng.choice(self.domains)
                chunks = chunks_by_domain[domain]

            # generate random chunk
            X, one_hot_y, labels = next(chunks)

            if (
                not self.osd
                or rng.random() > self.tasks["osd"].augmentation_probability
            ):
                # yield it as it is
                yield {"X": X, "y": self.prepare_y(one_hot_y)}
                continue

            #  generate second random chunk
            other_X, other_one_hot_y, other_labels = next(chunks)

            #  sum both chunks with random SNR
            random_snr = (
                self.tasks["osd"].snr_max - self.tasks["osd"].snr_min
            ) * rng.random() + self.tasks["osd"].snr_min
            alpha = np.exp(-np.log(10) * random_snr / 20)
            X = Audio.power_normalize(X) + alpha * Audio.power_normalize(other_X)

            # combine speaker-to-index mapping
            y_mapping = {label: i for i, label in enumerate(labels)}
            num_labels = len(y_mapping)
            for label in other_labels:
                if label not in y_mapping:
                    y_mapping[label] = num_labels
                    num_labels += 1

            #  combine one-hot-encoded speaker activities
            combined_y = np.zeros_like(one_hot_y, shape=(len(one_hot_y), num_labels))
            for i, label in enumerate(labels):
                combined_y[:, y_mapping[label]] += one_hot_y[:, i]
            for i, label in enumerate(other_labels):
                combined_y[:, y_mapping[label]] += other_one_hot_y[:, i]

            # handle corner case when the same label is active at the same time in both chunks
            combined_y = np.minimum(combined_y, 1, out=combined_y)

            yield {"X": X, "y": self.prepare_y(combined_y)}
