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
from typing import Iterable

import numpy as np
from pytorch_lightning import Callback, Trainer
from torch.nn import Parameter
from tqdm import tqdm

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.core.utils.distance import cdist, pdist
from pyannote.database.protocol import (
    SpeakerDiarizationProtocol,
    SpeakerVerificationProtocol,
)
from pyannote.metrics.binary_classification import det_curve


class SupervisedRepresentationLearningTaskMixin:
    """Methods common to most supervised representation tasks"""

    # batch_size = num_classes_per_batch x num_chunks_per_class

    @property
    def num_classes_per_batch(self) -> int:
        if hasattr(self, "num_classes_per_batch_"):
            return self.num_classes_per_batch_
        return self.batch_size // self.num_chunks_per_class

    @num_classes_per_batch.setter
    def num_classes_per_batch(self, num_classes_per_batch: int):
        self.num_classes_per_batch_ = num_classes_per_batch

    @property
    def num_chunks_per_class(self) -> int:
        if hasattr(self, "num_chunks_per_class_"):
            return self.num_chunks_per_class_
        return self.batch_size // self.num_classes_per_batch

    @num_chunks_per_class.setter
    def num_chunks_per_class(self, num_chunks_per_class: int):
        self.num_chunks_per_class_ = num_chunks_per_class

    @property
    def batch_size(self) -> int:
        if hasattr(self, "batch_size_"):
            return self.batch_size_
        return self.num_chunks_per_class * self.num_classes_per_batch

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.batch_size_ = batch_size

    def setup(self, stage=None):

        if stage == "fit":

            # loop over the training set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations, per class.

            self.train = dict()

            desc = f"Loading {self.protocol.name} training labels"
            for f in tqdm(iterable=self.protocol.train(), desc=desc, unit="file"):

                for klass in f["annotation"].labels():

                    # keep class's (long enough) speech turns...
                    speech_turns = [
                        segment
                        for segment in f["annotation"].label_timeline(klass)
                        if segment.duration > self.duration
                    ]

                    # skip if there is no speech turns left
                    if not speech_turns:
                        continue

                    # ... and their total duration
                    duration = sum(segment.duration for segment in speech_turns)

                    # add class to the list of classes
                    if klass not in self.train:
                        self.train[klass] = list()

                    self.train[klass].append(
                        {
                            "uri": f["uri"],
                            "audio": f["audio"],
                            "duration": duration,
                            "speech_turns": speech_turns,
                        }
                    )

            # there is no such thing as a "class" in representation
            # learning, so we do not need to define it here.
            self.specifications = TaskSpecification(
                problem=Problem.REPRESENTATION,
                scale=Scale.CHUNK,
                duration=self.duration,
                classes=sorted(self.train),
            )

            self.setup_loss_func()

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
        rng = create_rng_for_worker(self.current_epoch)

        classes = list(self.specifications.classes)

        while True:

            # shuffle classes so that we don't always have the same
            # groups of classes in a batch (which might be especially
            # problematic for contrast-based losses like contrastive
            # or triplet loss.
            rng.shuffle(classes)

            for klass in classes:

                # class index in original sorted order
                y = self.specifications.classes.index(klass)

                # multiple chunks per class
                for _ in range(self.num_chunks_per_class):

                    # select one file at random (with probability proportional to its class duration)
                    file, *_ = rng.choices(
                        self.train[klass],
                        weights=[f["duration"] for f in self.train[klass]],
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

                    X, _ = self.audio.crop(
                        file,
                        chunk,
                        mode="center",
                        fixed=self.duration,
                    )

                    yield {"X": X, "y": y}

    def train__len__(self):
        duration = sum(
            datum["duration"] for data in self.train.values() for datum in data
        )
        return math.ceil(duration / self.duration)

    def training_step(self, model: "Model", batch, batch_idx: int):

        X, y = batch["X"], batch["y"]
        loss = self.loss_func(model(X), y)

        model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def parameters(self, model: Model) -> Iterable[Parameter]:
        return chain(model.parameters(), self.loss_func.parameters())

    def val_dataloader(self):
        return None

    @property
    def val_monitor(self):
        return f"{self.ACRONYM}@train_loss", "min"

    def val_callback(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return _SpeakerVerificationValidationCallback(self)

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            return _SpeakerDiarizationValidationCallback(self)

        return None


class _SpeakerVerificationValidationCallback(Callback):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def on_epoch_end(self, trainer: Trainer, model: Model):

        inference = Inference(model, window="whole")

        y_true, y_pred = [], []
        emb = dict()
        # TODO: update tqdm desc
        for trial in tqdm(self.task.protocol.development_trial()):
            file1 = trial["file1"]
            uri1 = file1["uri"]
            file2 = trial["file2"]
            uri2 = file2["uri"]
            if uri1 not in emb:
                emb[uri1] = inference(file1).reshape(1, -1)
            if uri2 not in emb:
                emb[uri2] = inference(file2).reshape(1, -1)
            # TODO: make this "metric" thing an attribute
            y_pred.append(cdist(emb[uri1], emb[uri2], metric="cosine").item())
            y_true.append(trial["reference"])

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        fpr, fnr, thresholds, eer = det_curve(y_true, y_pred, distances=True)

        model.log(
            f"{self.task.ACRONYM}@val_eer",
            eer,
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )

        # TODO: log det curve

        # set model back to "train" mode
        model.train()

    @property
    def val_monitor(self):
        return f"{self.task.ACRONYM}@val_eer", "min"


class _SpeakerDiarizationValidationCallback(Callback):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def on_epoch_end(self, trainer: Trainer, model: Model):

        inference = Inference(model, window="whole")

        y_true, y_pred = [], []
        for file in tqdm(self.task.protocol.development()):
            X, y = zip(
                *[
                    (inference.crop(file, chunk), label)
                    for chunk, _, label in file["annotation"].itertracks(
                        yield_label=True
                    )
                ]
            )

            y_true.append(pdist(np.array(y), metric="equal"))
            y_pred.append(pdist(np.stack(X), metric="cosine"))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        fpr, fnr, thresholds, eer = det_curve(y_true, y_pred, distances=True)

        model.log(
            f"{self.task.ACRONYM}@val_eer",
            eer,
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )

        # TODO: log det curve

        # set model back to "train" mode
        model.train()

    @property
    def val_monitor(self):
        return f"{self.task.ACRONYM}@val_eer", "min"
