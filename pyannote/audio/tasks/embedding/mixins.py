# MIT License
#
# Copyright (c) 2020- CNRS
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
from typing import Dict, Sequence, Union

import torch
import torch.nn.functional as F
from pyannote.core import Segment
from pyannote.database.protocol import (
    SpeakerDiarizationProtocol,
    SpeakerVerificationProtocol,
)
from torch.utils.data._utils.collate import default_collate
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.torchmetrics.classification import EqualErrorRate
from pyannote.audio.utils.random import create_rng_for_worker


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

    def setup(self):
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
                    if segment.duration > self.min_duration
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
            min_duration=self.min_duration,
            classes=sorted(self._train),
        )

        if not self.has_validation:
            return

        if isinstance(self.protocol, SpeakerVerificationProtocol):
            self._validation = list(self.protocol.development_trial())

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        return [
            EqualErrorRate(compute_on_cpu=True, distances=False),
            BinaryAUROC(compute_on_cpu=True),
        ]

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

        # select batch-wise duration at random
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

                    # if speech turn is too short, pad with zeros
                    # TODO: handle this corner case with recently added mode="pad" option to audio.crop
                    if speech_turn.duration < batch_duration:
                        X, _ = self.model.audio.crop(file, speech_turn)
                        num_missing_frames = (
                            math.floor(batch_duration * self.model.audio.sample_rate)
                            - X.shape[1]
                        )
                        left_pad = rng.randint(0, num_missing_frames)
                        X = F.pad(X, (left_pad, num_missing_frames - left_pad))

                    # if it is long enough, select chunk at random
                    else:
                        start_time = rng.uniform(
                            speech_turn.start, speech_turn.end - batch_duration
                        )
                        chunk = Segment(start_time, start_time + batch_duration)

                        X, _ = self.model.audio.crop(
                            file,
                            chunk,
                            duration=batch_duration,
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
        return max(self.batch_size, math.ceil(duration / avg_chunk_duration))

    def collate_fn(self, batch, stage="train"):
        collated = default_collate(batch)

        if stage == "train":
            self.augmentation.train(mode=True)
            augmented = self.augmentation(
                samples=collated["X"],
                sample_rate=self.model.hparams.sample_rate,
            )
            collated["X"] = augmented.samples

        return collated

    def training_step(self, batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        loss = self.model.loss_func(self.model(X), y)

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None

        self.model.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return {"loss": loss}

    def val__getitem__(self, idx):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            trial = self._validation[idx]

            data = dict()
            for idx in [1, 2]:
                file = trial[f"file{idx:d}"]
                duration = self.model.audio.get_duration(file)
                if duration > self.duration:
                    middle = Segment(
                        0.5 * duration - 0.5 * self.duration,
                        0.5 * duration + 0.5 * self.duration,
                    )
                    X, _ = self.model.audio.crop(file, middle, duration=self.duration)
                else:
                    X, _ = self.model.audio(file)
                    num_missing_frames = (
                        math.floor(self.duration * self.model.audio.sample_rate)
                        - X.shape[1]
                    )
                    X = F.pad(X, (0, num_missing_frames))
                data[f"X{idx:d}"] = X
            data["y"] = trial["reference"]

            return data

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            pass

    def val__len__(self):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            return len(self._validation)

        elif isinstance(self.protocol, SpeakerDiarizationProtocol):
            return 0

    def validation_step(self, batch, batch_idx: int):
        if isinstance(self.protocol, SpeakerVerificationProtocol):
            with torch.no_grad():
                emb1 = self.model(batch["X1"]).detach()
                emb2 = self.model(batch["X2"]).detach()
                y_pred = F.cosine_similarity(emb1, emb2)

            y_true = batch["y"]
            self.model.validation_metric(y_pred, y_true)

            self.model.log_dict(
                self.model.validation_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
