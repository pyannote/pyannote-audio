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

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.task import Problem
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment, SlidingWindow


class SegmentationTaskMixin:
    """Methods common to most segmentation tasks"""

    def setup(self, stage=None):
        if stage == "fit":
            # loop over the training set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.
            self._train = []
            for f in self.protocol.train():
                segments = [
                    segment
                    for segment in f["annotated"]
                    if segment.duration > self.duration
                ]
                duration = sum(segment.duration for segment in segments)
                self._train.append(
                    {
                        "uri": f["uri"],
                        "annotated": segments,
                        "annotation": f["annotation"],
                        "duration": duration,
                        "audio": f["audio"],
                    }
                )

            # loop over the validation set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.
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
        return one_hot_y

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

        Returns
        -------
        X : np.ndarray
            Audio chunk as (num_samples, num_channels) array.
        y : np.ndarray, optional
            Frame-wise labels as (num_frames, num_labels) array.
        labels : list of str, optional
            Ordered labels such that y[:, k] corresponds to activity of labels[k].
        """

        X, _ = self.model.audio.crop(
            file,
            chunk,
            mode="center",
            fixed=self.duration if duration is None else duration,
        )

        introspection = self.model.introspection

        if self.is_multi_task:
            # this assumes that all tasks share the same model introspection.
            # this is a reasonable assumption for now.
            any_task = next(iter(introspection.keys()))
            num_frames, _ = introspection[any_task](X.shape[1])
        else:
            num_frames, _ = introspection(X.shape[1])

        annotation = file["annotation"].crop(chunk)
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
        y = np.minimum(y, 1, out=y)

        return X, y, labels

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
        rng = create_rng_for_worker(self.model.current_epoch)

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                self._train,
                weights=[f["duration"] for f in self._train],
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

            X, one_hot_y, _ = self.prepare_chunk(file, chunk, duration=self.duration)

            y = self.prepare_y(one_hot_y)

            yield {"X": X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self._train)
        return math.ceil(duration / self.duration)

    def val__getitem__(self, idx):
        f, chunk = self._validation[idx]
        X, one_hot_y, _ = self.prepare_chunk(f, chunk, duration=self.duration)
        y = self.prepare_y(one_hot_y)
        return {"X": X, "y": y}

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
            nrows=3 * nrows,
            ncols=ncols,
            figsize=(15, 10),
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
