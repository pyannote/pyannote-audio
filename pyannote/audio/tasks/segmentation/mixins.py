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
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from pyannote.database.protocol.protocol import Scope, Subset
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data._utils.collate import default_collate
from torchaudio import AudioMetaData
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC, MultilabelAUROC

from pyannote.audio.core.task import Problem, Task, get_dtype
from pyannote.audio.utils.random import create_rng_for_worker

Subsets = list(Subset.__args__)
Scopes = list(Scope.__args__)


class SegmentationTask(Task):
    """Methods common to most segmentation tasks"""

    def get_file(self, file_id):
        file = dict()

        file["audio"] = self.prepared_data["audio-path"][file_id]

        _audio_info = self.prepared_data["audio-info"][file_id]
        encoding = self.prepared_data["audio-encoding"][file_id]

        sample_rate = _audio_info["sample_rate"]
        num_frames = _audio_info["num_frames"]
        num_channels = _audio_info["num_channels"]
        bits_per_sample = _audio_info["bits_per_sample"]
        file["torchaudio.info"] = AudioMetaData(
            sample_rate=sample_rate,
            num_frames=num_frames,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            encoding=encoding,
        )

        return file

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

    def train__iter__helper(self, rng: random.Random, **filters):
        """Iterate over training samples with optional domain filtering

        Parameters
        ----------
        rng : random.Random
            Random number generator
        filters : dict, optional
            When provided (as {key: value} dict), filter training files so that
            only files such as file[key] == value are used for generating chunks.

        Yields
        ------
        chunk : dict
            Training chunks.
        """

        # indices of training files that matches domain filters
        training = self.prepared_data["audio-metadata"]["subset"] == Subsets.index(
            "train"
        )
        for key, value in filters.items():
            training &= self.prepared_data["audio-metadata"][key] == self.prepared_data[
                "metadata"
            ][key].index(value)
        file_ids = np.where(training)[0]

        # turn annotated duration into a probability distribution
        annotated_duration = self.prepared_data["audio-annotated"][file_ids]
        cum_prob_annotated_duration = np.cumsum(
            annotated_duration / np.sum(annotated_duration)
        )

        duration = self.duration

        num_chunks_per_file = getattr(self, "num_chunks_per_file", 1)

        while True:
            # select one file at random (with probability proportional to its annotated duration)
            file_id = file_ids[cum_prob_annotated_duration.searchsorted(rng.random())]

            # generate `num_chunks_per_file` chunks from this file
            for _ in range(num_chunks_per_file):
                # find indices of annotated regions in this file
                annotated_region_indices = np.where(
                    self.prepared_data["annotations-regions"]["file_id"] == file_id
                )[0]

                # turn annotated regions duration into a probability distribution

                cum_prob_annotated_regions_duration = np.cumsum(
                    self.prepared_data["annotations-regions"]["duration"][
                        annotated_region_indices
                    ]
                    / np.sum(
                        self.prepared_data["annotations-regions"]["duration"][
                            annotated_region_indices
                        ]
                    )
                )

                # selected one annotated region at random (with probability proportional to its duration)
                annotated_region_index = annotated_region_indices[
                    cum_prob_annotated_regions_duration.searchsorted(rng.random())
                ]

                # select one chunk at random in this annotated region
                _, region_duration, start = self.prepared_data["annotations-regions"][
                    annotated_region_index
                ]
                start_time = rng.uniform(start, start + region_duration - duration)

                yield self.prepare_chunk(file_id, start_time, duration)

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
        rng = create_rng_for_worker(self.model)

        balance = getattr(self, "balance", None)
        if balance is None:
            chunks = self.train__iter__helper(rng)

        else:
            # create a subchunk generator for each combination of "balance" keys
            subchunks = dict()
            for product in itertools.product(
                *[self.prepared_data["metadata"][key] for key in balance]
            ):
                # we iterate on the cartesian product of the values in metadata_unique_values
                # eg: for balance=["database", "split"], with 2 databases and 2 splits:
                # ("DIHARD", "A"), ("DIHARD", "B"), ("REPERE", "A"), ("REPERE", "B")
                filters = {key: value for key, value in zip(balance, product)}
                subchunks[product] = self.train__iter__helper(rng, **filters)

        while True:
            # select one subchunk generator at random (with uniform probability)
            # so that it is balanced on average
            if balance is not None:
                chunks = subchunks[rng.choice(list(subchunks))]

            # generate random chunk
            yield next(chunks)

    def collate_X(self, batch) -> torch.Tensor:
        return default_collate([b["X"] for b in batch])

    def collate_y(self, batch) -> torch.Tensor:
        return default_collate([b["y"].data for b in batch])

    def collate_meta(self, batch) -> torch.Tensor:
        return default_collate([b["meta"] for b in batch])

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

        # collate metadata
        collated_meta = self.collate_meta(batch)

        # apply augmentation (only in "train" stage)
        self.augmentation.train(mode=(stage == "train"))
        augmented = self.augmentation(
            samples=collated_X,
            sample_rate=self.model.hparams.sample_rate,
            targets=collated_y.unsqueeze(1),
        )

        return {
            "X": augmented.samples,
            "y": augmented.targets.squeeze(1),
            "meta": collated_meta,
        }

    def train__len__(self):
        # Number of training samples in one epoch
        train_file_ids = np.where(
            self.prepared_data["audio-metadata"]["subset"] == Subsets.index("train")
        )[0]

        duration = np.sum(self.prepared_data["audio-annotated"][train_file_ids])
        return max(self.batch_size, math.ceil(duration / self.duration))

    def prepare_validation(self, prepared_data: Dict):
        validation_chunks = list()

        # obtain indexes of files in the validation subset
        validation_file_ids = np.where(
            prepared_data["audio-metadata"]["subset"] == Subsets.index("development")
        )[0]

        # iterate over files in the validation subset
        for file_id in validation_file_ids:
            # get annotated regions in file
            annotated_regions = prepared_data["annotations-regions"][
                prepared_data["annotations-regions"]["file_id"] == file_id
            ]

            # iterate over annotated regions
            for annotated_region in annotated_regions:
                # number of chunks in annotated region
                num_chunks = round(annotated_region["duration"] // self.duration)

                # iterate over chunks
                for c in range(num_chunks):
                    start_time = annotated_region["start"] + c * self.duration
                    validation_chunks.append((file_id, start_time, self.duration))

        dtype = [
            (
                "file_id",
                get_dtype(max(v[0] for v in validation_chunks)),
            ),
            ("start", "f"),
            ("duration", "f"),
        ]

        prepared_data["validation"] = np.array(validation_chunks, dtype=dtype)
        validation_chunks.clear()

    def val__getitem__(self, idx):
        validation_chunk = self.prepared_data["validation"][idx]
        return self.prepare_chunk(
            validation_chunk["file_id"],
            validation_chunk["start"],
            duration=validation_chunk["duration"],
        )

    def val__len__(self):
        return len(self.prepared_data["validation"])

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
        y[y == 0] = np.nan
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

        for logger in self.model.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("samples", fig, self.model.current_epoch)
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"samples_epoch{self.model.current_epoch}.png",
                )

        plt.close(fig)
