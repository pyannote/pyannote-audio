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

import itertools
import math
import random
import warnings
from typing import Dict, Optional, Sequence, Text, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from pyannote.core import Segment, SlidingWindowFeature
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from torch.utils.data._utils.collate import default_collate
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MultilabelAUROC, MulticlassAUROC

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.task import Problem
from pyannote.audio.utils.random import create_rng_for_worker


class SegmentationTaskMixin:
    """Methods common to most segmentation tasks"""

    def setup(self, stage: Optional[str] = None):

        # ==================================================================
        # PREPARE TRAINING DATA
        # ==================================================================

        self._train = []
        self._train_metadata = dict()

        for f in self.protocol.train():

            file = dict()

            for key, value in f.items():

                # keep track of unique labels in self._train_metadata["annotation"]
                if key == "annotation":
                    for label in value.labels():
                        self._train_metadata.setdefault("annotation", set()).add(label)

                # pass "audio" entry as it is
                elif key == "audio":
                    pass

                # remove segments shorter than chunks from "annotated" entry
                elif key == "annotated":
                    value = [
                        segment for segment in value if segment.duration > self.duration
                    ]
                    file["_annotated_duration"] = sum(
                        segment.duration for segment in value
                    )

                # keey track of unique text-like entries (incl. "uri" and "database")
                # and pass them as they are
                elif isinstance(value, Text):
                    self._train_metadata.setdefault(key, set()).add(value)

                # pass score-like entries as they are
                elif isinstance(value, SlidingWindowFeature):
                    pass

                else:
                    msg = (
                        f"Protocol '{self.protocol.name}' defines a '{key}' entry of type {type(value)} "
                        f"which we do not know how to handle."
                    )
                    warnings.warn(msg)

                file[key] = value

            self._train.append(file)

        self._train_metadata = {
            key: sorted(values) for key, values in self._train_metadata.items()
        }

        # ==================================================================
        # PREPARE VALIDATION DATA
        # ==================================================================

        if not self.has_validation:
            return

        self._validation = []

        for f in self.protocol.development():

            for segment in f["annotated"]:

                if segment.duration < self.duration:
                    continue

                num_chunks = round(segment.duration // self.duration)

                for c in range(num_chunks):
                    start_time = segment.start + c * self.duration
                    chunk = Segment(start_time, start_time + self.duration)
                    self._validation.append((f, chunk))

        random.shuffle(self._validation)

    def default_metric(
        self,
    ) -> Union[Metric, Sequence[Metric], Dict[str, Metric]]:
        """Returns macro-average of the area under the ROC curve"""

        num_classes = len(self.specifications.classes)
        if self.specifications.problem == Problem.BINARY_CLASSIFICATION:
            return BinaryAUROC(compute_on_cpu=True)
        elif self.specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return MultilabelAUROC(num_classes, average="macro", compute_on_cpu=True)
        elif self.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return MulticlassAUROC(num_classes, average="macro", compute_on_cpu=True)
        else:
            raise RuntimeError(
                f"The {self.specifications.problem} problem type hasn't been given a default segmentation metric yet."
            )

    def adapt_y(self, one_hot_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `adapt_y` method."
        )

    def prepare_chunk(
        self,
        file: AudioFile,
        chunk: Segment,
        duration: float = None,
    ) -> dict:
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
        sample : dict
            Dictionary with the following keys:
            X : np.ndarray
                Audio chunk as (num_samples, num_channels) array.
            y : SlidingWindowFeature
                Frame-wise labels as (num_frames, num_labels) array.

        """

        sample = dict()

        # read (and resample if needed) audio chunk
        duration = duration or self.duration
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # use model introspection to predict how many frames it will output
        num_samples = sample["X"].shape[1]
        num_frames, _ = self.model.introspection(num_samples)
        resolution = duration / num_frames

        # discretize annotation, using model resolution
        sample["y"] = file["annotation"].discretize(
            support=chunk, resolution=resolution, duration=duration
        )

        return sample

    def train__iter__helper(self, rng: random.Random, **domain_filter):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        domain_filter : dict, optional
            When provided (as {domain_key: domain_value} dict), filter training files so that
            only files such as file[domain_key] == domain_value are used for generating chunks.

        Yields
        ------
        chunk : dict
            Training chunks.
        """

        train = self._train

        try:
            domain_key, domain_value = domain_filter.popitem()
        except KeyError:
            domain_key = None

        if domain_key is not None:
            train = [f for f in train if f[domain_key] == domain_value]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                train,
                weights=[f["_annotated_duration"] for f in train],
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
        dict:
            X: (time, channel)
                Audio chunks.
            y: (frame, )
                Frame-level targets. Note that frame < time.
                `frame` is infered automagically from the
                example model output.
            ...
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker(self.model.current_epoch)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)

        else:
            chunks_by_domain = {
                domain: self.train__iter__helper(rng, **{balance: domain})
                for domain in self._train_metadata[balance]
            }

        while True:

            if balance is not None:
                domain = rng.choice(self._train_metadata[balance])
                chunks = chunks_by_domain[domain]

            # generate random chunk
            yield next(chunks)

    def collate_X(self, batch) -> torch.Tensor:
        return default_collate([b["X"] for b in batch])

    def collate_y(self, batch) -> torch.Tensor:

        # gather common set of labels
        # b["y"] is a SlidingWindowFeature instance
        labels = sorted(set(itertools.chain(*(b["y"].labels for b in batch))))

        batch_size, num_frames, num_labels = (
            len(batch),
            len(batch[0]["y"]),
            len(labels),
        )
        Y = np.zeros((batch_size, num_frames, num_labels), dtype=np.int64)

        for i, b in enumerate(batch):
            for local_idx, label in enumerate(b["y"].labels):
                global_idx = labels.index(label)
                Y[i, :, global_idx] = b["y"].data[:, local_idx]

        return torch.from_numpy(Y)

    def adapt_y(self, collated_y: torch.Tensor) -> torch.Tensor:
        return collated_y

    def collate_fn(self, batch, stage="train"):
        """Collate function used for most segmentation tasks

        This function does the following:
        * stack waveforms into a (batch_size, num_channels, num_samples) tensor batch["X"])
        * apply augmentation when in "train" stage
        * convert targets into a (batch_size, num_frames, num_classes) tensor batch["y"]
        * collate any other keys that might be present in the batch using pytorch default_collate function

        Parameters
        ----------
        batch : list of dict
            List of training samples.

        Returns
        -------
        batch : dict
            Collated batch as {"X": torch.Tensor, "y": torch.Tensor} dict.
        """

        # collate X
        collated_X = self.collate_X(batch)

        # collate y
        collated_y = self.collate_y(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y.unsqueeze(1),
        )

        return {"X": augmented.samples, "y": self.adapt_y(augmented.targets.squeeze(1))}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["_annotated_duration"] for file in self._train)
        return max(self.batch_size, math.ceil(duration / self.duration))

    def val__getitem__(self, idx):
        f, chunk = self._validation[idx]
        return self.prepare_chunk(f, chunk, duration=self.duration)

    def val__len__(self):
        return len(self._validation)

    def validation_step(self, batch, batch_idx: int):
        """Compute validation area under the ROC curve

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        X, y = batch["X"], batch["y"]
        # X = (batch_size, num_channels, num_samples)
        # y = (batch_size, num_frames, num_classes) or (batch_size, num_frames)

        y_pred = self.model(X)
        _, num_frames, _ = y_pred.shape
        # y_pred = (batch_size, num_frames, num_classes)

        # - remove warm-up frames
        # - downsample remaining frames
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        preds = y_pred[:, warm_up_left : num_frames - warm_up_right : 10]
        target = y[:, warm_up_left : num_frames - warm_up_right : 10]

        # torchmetrics tries to be smart about the type of machine learning problem
        # pyannote.audio is more explicit so we have to reshape target and preds for
        # torchmetrics to be happy... more details can be found here:
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#input-types

        if self.specifications.problem == Problem.BINARY_CLASSIFICATION:
            # target: shape (batch_size, num_frames), type binary
            # preds:  shape (batch_size, num_frames, 1), type float

            # torchmetrics expects:
            # target: shape (batch_size,), type binary
            # preds:  shape (batch_size,), type float

            self.model.validation_metric(
                preds.reshape(-1),
                target.reshape(-1),
            )

        elif self.specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            # target: shape (batch_size, num_frames, num_classes), type binary
            # preds:  shape (batch_size, num_frames, num_classes), type float

            # torchmetrics expects
            # target: shape (batch_size, num_classes, ...), type binary
            # preds:  shape (batch_size, num_classes, ...), type float

            self.model.validation_metric(
                torch.transpose(preds, 1, 2),
                torch.transpose(target, 1, 2),
            )

        elif self.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            # TODO: implement when pyannote.audio gets its first mono-label segmentation task
            raise NotImplementedError()

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # log first batch visualization every 2^n epochs.
        if (
            self.model.current_epoch == 0
            or math.log2(self.model.current_epoch) % 1 > 0
            or batch_idx > 0
        ):
            return

        # visualize first 9 validation samples of first batch in Tensorboard/MLflow
        X = X.cpu().numpy()
        y = y.float().cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.NaN
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):

            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot target
            ax_ref = axes[row_idx * 2 + 0, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot predictions
            ax_hyp = axes[row_idx * 2 + 1, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.axvspan(0, warm_up_left, color="k", alpha=0.5, lw=0)
            ax_hyp.axvspan(
                num_frames - warm_up_right, num_frames, color="k", alpha=0.5, lw=0
            )
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

        plt.tight_layout()

        if isinstance(self.model.logger, TensorBoardLogger):
            self.model.logger.experiment.add_figure(
                f"{self.logging_prefix}ValSamples", fig, self.model.current_epoch
            )
        elif isinstance(self.model.logger, MLFlowLogger):
            self.model.logger.experiment.log_figure(
                run_id=self.model.logger.run_id,
                figure=fig,
                artifact_file=f"{self.logging_prefix}ValSamples_epoch{self.model.current_epoch}.png",
            )

        plt.close(fig)
