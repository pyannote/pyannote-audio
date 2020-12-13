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
from typing import Optional

import numpy as np
from pytorch_lightning import Callback, Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification, ValDataset
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

            if isinstance(self.protocol, SpeakerVerificationProtocol):
                sessions = dict()
                for trial in self.protocol.development_trial():
                    for session in ["file1", "file2"]:
                        session_hash = hash(tuple(trial[session]))
                        if session_hash not in sessions:
                            sessions[session_hash] = trial[session]
                self.validation = sessions

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

        batch_duration = rng.uniform(self.min_duration, self.duration)
        num_samples = 0

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
                        speech_turn.start, speech_turn.end - batch_duration
                    )
                    chunk = Segment(start_time, start_time + batch_duration)

                    X, _ = self.audio.crop(
                        file,
                        chunk,
                        mode="center",
                        fixed=batch_duration,
                    )

                    yield {"X": X, "y": y}

                    num_samples += 1
                    if num_samples == self.batch_size:
                        batch_duration = rng.uniform(self.min_duration, self.duration)
                        num_samples = 0

    def train__len__(self):
        duration = sum(
            datum["duration"] for data in self.train.values() for datum in data
        )
        avg_chunk_duration = 0.5 * (self.min_duration + self.duration)
        return math.ceil(duration / avg_chunk_duration)

    def training_step(self, model: "Model", batch, batch_idx: int):

        X, y = batch["X"], batch["y"]
        loss = model.loss_func(model(X), y)

        model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    def val__iter__(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            for session_hash, session in self.validation.items():
                X = np.concatenate(
                    [
                        self.audio.crop(session, segment, mode="center")[0]
                        for segment in session["try_with"]
                    ],
                    axis=0,
                )
                yield {"session_hash": session_hash, "X": X}

    def val__len__(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return len(self.validation)

    def validation_step(self, model: "Model", batch, batch_idx: int):

        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return batch["session_hash"][0], model(batch["X"]).detach().cpu().numpy()

        else:
            raise NotImplementedError("")

    def validation_epoch_end(self, model: "Model", outputs):

        if isinstance(self.protocol, SpeakerVerificationProtocol):

            embeddings = dict(outputs)

            y_true, y_pred = [], []
            for trial in self.protocol.development_trial():

                session1_hash = hash(tuple(trial["file1"]))
                session2_hash = hash(tuple(trial["file2"]))

                try:
                    emb1 = embeddings[session1_hash]
                    emb2 = embeddings[session2_hash]
                except KeyError:
                    return

                y_pred.append(cdist(emb1, emb2, metric="cosine").item())
                y_true.append(trial["reference"])

            fpr, fnr, thresholds, eer = det_curve(
                np.array(y_true), np.array(y_pred), distances=True
            )

            model.log(
                f"{self.task.ACRONYM}@val_eer",
                eer,
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )

    def val_dataloader(self) -> Optional[DataLoader]:

        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return DataLoader(
                ValDataset(self),
                batch_size=1,
                # num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )

        return None

    def val_callback(self) -> Optional[Callback]:

        if isinstance(self.protocol, SpeakerDiarizationProtocol):
            return _SpeakerDiarizationValidationCallback(self)

        return None

    @property
    def val_monitor(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return f"{self.ACRONYM}@val_eer", "min"

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
