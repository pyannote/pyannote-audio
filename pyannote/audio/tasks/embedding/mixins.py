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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyannote.audio.core.task import Problem, Resolution, Specifications, ValDataset
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.core.utils.distance import cdist
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

            self._train = dict()

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
                    if klass not in self._train:
                        self._train[klass] = list()

                    self._train[klass].append(
                        {
                            "uri": f["uri"],
                            "audio": f["audio"],
                            "duration": duration,
                            "speech_turns": speech_turns,
                        }
                    )

            self.specifications = Specifications(
                problem=Problem.REPRESENTATION,
                resolution=Resolution.CHUNK,
                duration=self.duration,
                classes=sorted(self._train),
            )

            if not self.has_validation:
                return

            if isinstance(self.protocol, SpeakerVerificationProtocol):
                sessions = dict()
                for trial in self.protocol.development_trial():
                    for session in ["file1", "file2"]:
                        session_hash = self.helper_trial_hash(trial[session])
                        if session_hash not in sessions:
                            sessions[session_hash] = trial[session]
                self._validation = list(sessions.items())

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
        rng = create_rng_for_worker(self.model.current_epoch)

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
                        self._train[klass],
                        weights=[f["duration"] for f in self._train[klass]],
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

                    X, _ = self.model.audio.crop(
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
            datum["duration"] for data in self._train.values() for datum in data
        )
        avg_chunk_duration = 0.5 * (self.min_duration + self.duration)
        return math.ceil(duration / avg_chunk_duration)

    def training_step(self, batch, batch_idx: int):

        X, y = batch["X"], batch["y"]
        loss = self.model.loss_func(self.model(X), y)

        self.model.log(
            f"{self.ACRONYM}@train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss}

    @staticmethod
    def helper_trial_hash(file) -> int:
        return hash((file["database"], file["uri"], tuple(file["try_with"])))

    def val__getitem__(self, idx):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            session_hash, session = self._validation[idx]
            X = np.concatenate(
                [
                    self.model.audio.crop(session, segment, mode="center")[0]
                    for segment in session["try_with"]
                ],
                axis=0,
            )
            return {"session_hash": session_hash, "X": X}

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            pass

    def val__len__(self):

        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return len(self._validation)

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            return 0

    def validation_step(self, batch, batch_idx: int):

        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return (
                batch["session_hash"][0].detach().cpu().numpy().item(),
                self.model(batch["X"]).detach().cpu().numpy(),
            )

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            pass

    def validation_epoch_end(self, outputs):

        if isinstance(self.protocol, SpeakerVerificationProtocol):

            embeddings = dict(outputs)

            y_true, y_pred = [], []
            for trial in self.protocol.development_trial():

                session1_hash = self.helper_trial_hash(trial["file1"])
                session2_hash = self.helper_trial_hash(trial["file2"])

                try:
                    emb1 = embeddings[session1_hash]
                    emb2 = embeddings[session2_hash]
                except KeyError:
                    return

                y_pred.append(cdist(emb1, emb2, metric="cosine").item())
                y_true.append(trial["reference"])

            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            num_target_trials = np.sum(y_true)
            num_non_target_trials = np.sum(1.0 - y_true)
            if num_target_trials > 2 and num_non_target_trials > 2:
                fpr, fnr, thresholds, eer = det_curve(y_true, y_pred, distances=True)

                self.model.log(
                    f"{self.ACRONYM}@val_eer",
                    torch.tensor(eer, device=self.model.device),
                    logger=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            pass

    def val_dataloader(self) -> Optional[DataLoader]:

        if self.has_validation:

            if isinstance(self.protocol, SpeakerVerificationProtocol):
                return DataLoader(
                    ValDataset(self),
                    batch_size=1,
                    pin_memory=self.pin_memory,
                    drop_last=False,
                )

            elif isinstance(self.protocol, SpeakerDiarizationProtocol):
                return None

        else:
            return None

    @property
    def val_monitor(self):

        if self.has_validation:

            if isinstance(self.protocol, SpeakerVerificationProtocol):
                return f"{self.ACRONYM}@val_eer", "min"

            elif isinstance(self.protocol, SpeakerDiarizationProtocol):
                return None, "min"

        else:
            return None, "min"
