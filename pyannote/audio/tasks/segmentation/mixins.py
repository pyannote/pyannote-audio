# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
import random
import warnings
from typing import List, Optional, Text, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.metrics.classification import FBeta
from typing_extensions import Literal

from pyannote.audio.core.io import Audio, AudioFile
from pyannote.audio.core.task import Problem
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature


class SegmentationTaskMixin:
    """Methods common to most segmentation tasks"""

    def setup(self, stage=None):

        if stage == "fit":

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
                            self._train_metadata.setdefault("annotation", set()).add(
                                label
                            )

                    # pass "audio" entry as it is
                    elif key == "audio":
                        pass

                    # remove segments shorter than chunks from "annotated" entry
                    elif key == "annotated":
                        value = [
                            segment
                            for segment in value
                            if segment.duration > self.duration
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

    def setup_validation_metric(self):
        """Setup default validation metric

        Use macro-average of F-score with a 0.5 threshold
        """

        self.val_fbeta = FBeta(
            len(self.specifications.classes),
            beta=1.0,
            threshold=0.5,
            multilabel=(
                self.specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION
            ),
            average="macro",
        )

    def prepare_y(self, one_hot_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the `prepare_y` method."
        )

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
        stage: Literal["train", "val"] = "train",
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
        stage : {"train", "val"}
            "train" for training step, "val" for validation step

        Returns
        -------
        sample : dict
            Dictionary with the following keys:
            X : np.ndarray
                Audio chunk as (num_samples, num_channels) array.
            y : np.ndarray
                Frame-wise labels as (num_frames, num_labels) array.
            ...
        """

        sample = dict()

        # ==================================================================
        # X = "audio" crop
        # ==================================================================

        sample["X"], _ = self.model.audio.crop(
            file,
            chunk,
            mode="center",
            fixed=self.duration if duration is None else duration,
        )

        # ==================================================================
        # y = "annotation" crop (with corresponding "labels")
        # ==================================================================

        # use model introspection to predict how many frames it will output
        num_samples = sample["X"].shape[1]
        introspection = self.model.introspection
        if self.is_multi_task:
            # this assumes that all tasks share the same model introspection.
            # this is a reasonable assumption for now.
            any_task = next(iter(introspection.keys()))
            num_frames, _ = introspection[any_task](num_samples)
        else:
            num_frames, _ = introspection(num_samples)

        # crop "annotation" and keep track of corresponding list of labels if needed
        annotation: Annotation = file["annotation"].crop(chunk)
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
        sample["y"] = np.minimum(y, 1, out=y)
        sample["labels"] = labels

        # ==================================================================
        # additional metadata
        # ==================================================================

        for key, value in file.items():

            # those keys were already dealt with
            if key in ["audio", "annotation", "annotated"]:
                pass

            # replace text-like entries by their integer index
            elif isinstance(value, Text):
                try:
                    sample[key] = self._train_metadata[key].index(value)
                except ValueError as e:
                    if stage == "val":
                        sample[key] = -1
                    else:
                        raise e

            # crop score-like entries
            elif isinstance(value, SlidingWindowFeature):
                sample[key] = value.crop(chunk, fixed=duration, mode="center")

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

            yield self.prepare_chunk(file, chunk, duration=self.duration, stage="train")

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
        overlap = getattr(self, "overlap", dict())
        overlap_probability = overlap.get("probability", 0.0)
        if overlap_probability > 0:
            overlap_snr_min = overlap.get("snr_min", 0.0)
            overlap_snr_max = overlap.get("snr_max", 0.0)

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
            sample = next(chunks)

            if rng.random() > overlap_probability:
                try:
                    sample["y"] = self.prepare_y(sample["y"])
                except ValueError:
                    # if a ValueError is raised by prepare_y, skip this sample.

                    # see pyannote.audio.tasks.segmentation.Segmentation.prepare_y
                    # to understand why this might happen.
                    continue

                _ = sample.pop("labels")
                yield sample
                continue

            # generate another random chunk
            other_sample = next(chunks)

            # sum both chunks with random SNR
            random_snr = (
                overlap_snr_max - overlap_snr_min
            ) * rng.random() + overlap_snr_min
            alpha = np.exp(-np.log(10) * random_snr / 20)
            combined_X = Audio.power_normalize(
                sample["X"]
            ) + alpha * Audio.power_normalize(other_sample["X"])

            # combine labels
            y, labels = sample["y"], sample.pop("labels")
            other_y, other_labels = other_sample["y"], other_sample.pop("labels")
            y_mapping = {label: i for i, label in enumerate(labels)}
            num_combined_labels = len(y_mapping)
            for label in other_labels:
                if label not in y_mapping:
                    y_mapping[label] = num_combined_labels
                    num_combined_labels += 1
            # combined_labels = [
            #     label
            #     for label, _ in sorted(y_mapping.items(), key=lambda item: item[1])
            # ]

            # combine targets
            combined_y = np.zeros_like(y, shape=(len(y), num_combined_labels))
            for i, label in enumerate(labels):
                combined_y[:, y_mapping[label]] += y[:, i]
            for i, label in enumerate(other_labels):
                combined_y[:, y_mapping[label]] += other_y[:, i]

            # handle corner case when the same label is active at the same time in both chunks
            combined_y = np.minimum(combined_y, 1, out=combined_y)

            try:
                combined_y = self.prepare_y(combined_y)
            except ValueError:
                # if a ValueError is raised by prepare_y, skip this sample.

                # see pyannote.audio.tasks.segmentation.Segmentation.prepare_y
                # to understand why this might happen.
                continue

            combined_sample = {
                "X": combined_X,
                "y": combined_y,
            }

            for key, value in sample.items():

                # those keys were already dealt with
                if key in ["X", "y"]:
                    pass

                # text-like entries have been replaced by their integer index in prepare_chunk.
                # we (somewhat arbitrarily) combine i and j into i + j x (num_values + 1) to avoid
                # any conflict with pure i or pure j samples
                elif isinstance(value, int):
                    combined_sample[key] = sample[key] + other_sample[key] * (
                        len(self._train_metadata[key]) + 1
                    )

                # score-like entries have been chunked into numpy array in prepare_chunk
                # we (somewhat arbitrarily) average them using the same alpha as for X
                elif isinstance(value, np.ndarray):
                    combined_sample[key] = (sample[key] + alpha * other_sample[key]) / (
                        1 + alpha
                    )

            yield combined_sample

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["_annotated_duration"] for file in self._train)
        return max(self.batch_size, math.ceil(duration / self.duration))

    def val__getitem__(self, idx):
        f, chunk = self._validation[idx]
        sample = self.prepare_chunk(f, chunk, duration=self.duration, stage="val")
        sample["y"] = self.prepare_y(sample["y"])
        _ = sample.pop("labels")
        return sample

    def val__len__(self):
        return len(self._validation)

    def validation_step(self, batch, batch_idx: int):
        """Compute area under ROC curve

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        # move metric to model device
        self.val_fbeta.to(self.model.device)

        X, y = batch["X"], batch["y"]
        # X = (batch_size, num_channels, num_samples)
        # y = (batch_size, num_frames, num_classes)

        y_pred = self.model(X)
        # y_pred = (batch_size, num_frames, num_classes)

        val_fbeta = self.val_fbeta(y_pred[:, ::10].squeeze(), y[:, ::10].squeeze())
        self.model.log(
            f"{self.ACRONYM}@val_fbeta",
            val_fbeta,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_idx > 0:
            return

        # visualize first 9 validation samples of first batch in Tensorboard
        X = X.cpu().numpy()
        y = y.float().cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        # prepare 3 x 3 grid (or smaller if batch size is smaller)
        num_samples = min(self.batch_size, 9)
        nrows = math.ceil(math.sqrt(num_samples))
        ncols = math.ceil(num_samples / nrows)
        fig, axes = plt.subplots(
            nrows=3 * nrows, ncols=ncols, figsize=(15, 10), squeeze=False
        )

        # reshape target so that there is one line per class when plottingit
        y[y == 0] = np.NaN
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):

            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot waveform
            ax_wav = axes[row_idx * 3 + 0, col_idx]
            sample_X = np.mean(X[sample_idx], axis=0)
            ax_wav.plot(sample_X)
            ax_wav.set_xlim(0, len(sample_X))
            ax_wav.get_xaxis().set_visible(False)
            ax_wav.get_yaxis().set_visible(False)

            # plot target
            ax_ref = axes[row_idx * 3 + 1, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot prediction
            ax_hyp = axes[row_idx * 3 + 2, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

        plt.tight_layout()

        self.model.logger.experiment.add_figure(
            f"{self.ACRONYM}@val_samples", fig, self.model.current_epoch
        )

    @property
    def val_monitor(self):
        """Maximize validation area under ROC curve"""
        return f"{self.ACRONYM}@val_fbeta", "max"
