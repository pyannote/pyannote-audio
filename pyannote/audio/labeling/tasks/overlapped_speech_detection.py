#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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

import numpy as np
import random
import math
from typing import List, Dict, Union, Text
from tqdm import trange, tqdm

from torch.utils.data import IterableDataset

from pyannote.core import Segment, Timeline
from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.audio.features import RawAudio

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

from argparse import Namespace

from pyannote.audio.pipeline import OverlapDetection as OverlapDetectionPipeline
from pyannote.pipeline import Optimizer
from pyannote.database import ProtocolFile
from pyannote.database import Protocol
from pyannote.database import Subset

from pyannote.audio.features.utils import normalize


class Dataset(IterableDataset):
    def __init__(self, task: BaseTask):
        super().__init__()
        self.task = task
        self.raw_audio_ = RawAudio(sample_rate=self.task.feature_extraction.sample_rate)

    def __iter__(self):
        random.seed()

        if self.task.hparams.domain is None:
            chunks = self.chunks()
        else:
            chunks_by_domain = {
                domain: self.chunks(domain=domain)
                for domain in self.task.train_metadata["domains"]
            }

        while True:

            if self.task.hparams.domain:
                #  draw domain at random
                domain = random.choice(self.task.train_metadata["domains"])
                chunks = chunks_by_domain[domain]

            chunk = next(chunks)
            if random.random() > 0.5:
                waveform = chunk["X"]
                y = chunk["y"]

            else:
                other_chunk = next(chunks)

                # combine both waveforms
                waveform = chunk["X"]
                other_waveform = other_chunk["X"]
                random_snr = (
                    self.task.hparams.snr_max - self.task.hparams.snr_min
                ) * random.random() + self.task.hparams.snr_min
                alpha = np.exp(-np.log(10) * random_snr / 20)
                waveform = normalize(waveform) + alpha * normalize(other_waveform)

                y_mapping = {label: i for i, label in enumerate(chunk["labels"])}
                num_labels = len(y_mapping)
                for label in other_chunk["labels"]:
                    if label not in y_mapping:
                        y_mapping[label] = num_labels
                        num_labels += 1

                y = chunk["y"]
                other_y = other_chunk["y"]
                combined_y = np.zeros_like(y, shape=(len(y), num_labels))
                for i, label in enumerate(chunk["labels"]):
                    combined_y[:, y_mapping[label]] += y[:, i]
                for i, label in enumerate(other_chunk["labels"]):
                    combined_y[:, y_mapping[label]] += other_y[:, i]
                y = combined_y

            X = self.task.feature_extraction.crop(
                {"waveform": waveform, "duration": self.task.hparams.duration},
                Segment(0, self.task.hparams.duration),
                mode="center",
                fixed=self.task.hparams.duration,
            )
            y = np.int64(np.sum(y, axis=1, keepdims=True) > 1)

            yield {"X": X, "y": y}

    def chunks(self, domain: Text = None):

        files = self.task.train_metadata["files"]

        if domain is not None:
            files = [f for f in files if f[self.task.hparams.domain] == domain]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = random.choices(
                files, weights=[f["__duration"] for f in files], k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = random.choices(
                file["annotated"], weights=[s.duration for s in file["annotated"]], k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = random.uniform(
                segment.start, segment.end - self.task.hparams.duration
            )
            chunk = Segment(start_time, start_time + self.task.hparams.duration)

            # extract waveform
            X = self.raw_audio_.crop(
                file, chunk, mode="center", fixed=self.task.hparams.duration
            )

            # extract target
            y = file["__target"].crop(
                chunk, mode=self.task.model.alignment, fixed=self.task.hparams.duration
            )

            # yield sample
            yield {"X": X, "y": y, "labels": file["__target"].labels}

    def __len__(self):

        epoch_duration = self.task.train_metadata["epoch_duration"]

        num_samples = math.ceil(epoch_duration / self.task.hparams.duration)

        # TODO: remove when https://github.com/pytorch/pytorch/pull/38925 is released
        num_samples = max(1, num_samples // self.task.hparams.batch_size)

        return num_samples


class OverlappedSpeechDetection(BaseTask):
    """Overlapped speech detection"""

    problem = Problem.MULTI_CLASS_CLASSIFICATION
    resolution_input = Resolution.FRAME
    resolution_output = Resolution.FRAME

    def __init__(
        self, hparams: Union[Namespace, Dict], **kwargs,
    ):

        super().__init__(hparams, **kwargs)
        if "snr_min" not in self.hparams:
            self.hparams.snr_min = 0.0
        if "snr_max" not in self.hparams:
            self.hparams.snr_max = 10.0
        if "domain" not in self.hparams:
            self.hparams.domain = None

    def prepare_metadata(self, files: List[ProtocolFile]) -> Dict:

        output_resolution = self.model.get_resolution()

        for f in tqdm(iterable=files, desc="Loading training metadata", unit="file"):

            f["__duration"] = sum(
                s.duration for s in f["annotated"] if s.duration > self.hparams.duration
            )

            y = one_hot_encoding(
                f["annotation"],
                Timeline(segments=[Segment(0, f["duration"])]),
                output_resolution,
                mode="center",
            )

            f["__target"] = y
            del f["annotation"]

        metadata = {
            "classes": ["non_overlap", "overlap"],
            "epoch_duration": sum(f["__duration"] for f in files),
            "files": [dict(f) for f in files],
        }

        if self.hparams.domain is not None:
            metadata["domains"] = list(set(f[self.hparams.domain] for f in files))

        return metadata

    def train_dataset(self) -> IterableDataset:
        return Dataset(self)

    @staticmethod
    def validation_criterion(protocol: Protocol):
        return "detection_fscore"

    def validation_pipeline(self):
        pipeline = OverlapDetectionPipeline(
            scores="@scores", fscore=True, hysteresis=False
        )
        pipeline.freeze(
            {
                "min_duration_on": 0.1,
                "min_duration_off": 0.1,
                "pad_onset": 0.0,
                "pad_offset": 0.0,
            }
        )
        return pipeline

    def validation(
        self,
        files: List[ProtocolFile],
        protocol: Protocol = None,
        subset: Subset = "development",
        warm_start: Dict = None,
        epoch: int = None,
    ):
        """Validation

        Validation consists in looking for the value of the detection threshold
        that maximizes the f-score of recall and precision.
        """

        criterion = self.validation_criterion(protocol)
        pipeline = self.validation_pipeline()

        show_progress = {"unit": "file", "leave": False, "position": 2}

        optimizer = Optimizer(pipeline)
        iterations = optimizer.tune_iter(
            files, warm_start=warm_start, show_progress=show_progress
        )

        for i in trange(
            10,
            unit="iteration",
            position=1,
            leave=False,
            desc=f"epoch #{epoch} | optimizing threshold...",
        ):
            result = next(iterations)

        return {
            "metric": criterion,
            "minimize": False,
            "value": result["loss"],
            "params": result["params"],
        }
