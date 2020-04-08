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

import torch
import torch.nn.functional as F
from pyannote.audio.train.trainer import Trainer
import numpy as np

from typing import Text
from typing import Optional
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.audio.features import FeatureExtraction
from pyannote.database.protocol.protocol import Protocol
from pyannote.audio.features.wrapper import Wrappable
from pyannote.audio.train.task import Task, TaskType, TaskOutput


class RepresentationLearning(Trainer):
    """

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    min_duration : float, optional
        When provided, use chunks of random duration between `min_duration` and
        `duration` for training. Defaults to using fixed duration chunks.
    per_turn : int, optional
        Number of chunks per speech turn. Defaults to 1.
        If per_turn is greater than one, embeddings of the same speech turn
        are averaged before classification. The intuition is that it might
        help learn embeddings meant to be averaged/summed.
    per_label : `int`, optional
        Number of sequences per speaker in each batch. Defaults to 1.
    per_fold : `int`, optional
        Number of different speakers per batch. Defaults to 32.
    per_epoch : `float`, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : `float`, optional
        Remove speakers with less than that many seconds of speech.
        Defaults to 0 (i.e. keep them all).
    """

    def __init__(
        self,
        duration: float = 1.0,
        min_duration: float = None,
        per_turn: int = 1,
        per_label: int = 1,
        per_fold: Optional[int] = None,
        per_epoch: Optional[float] = None,
        label_min_duration: float = 0.0,
    ):

        super().__init__()
        self.duration = duration
        self.min_duration = min_duration
        self.per_turn = per_turn
        self.per_label = per_label
        self.per_fold = per_fold
        self.per_epoch = per_epoch
        self.label_min_duration = label_min_duration

    def get_batch_generator(
        self,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Text = "train",
        **kwargs
    ) -> SpeechSegmentGenerator:
        """Get batch generator

        Parameters
        ----------
        feature_extraction : `FeatureExtraction`
        protocol : `Protocol`
        subset : {'train', 'development', 'test'}, optional

        Returns
        -------
        generator : `SpeechSegmentGenerator`
        """

        return SpeechSegmentGenerator(
            feature_extraction,
            protocol,
            subset=subset,
            duration=self.duration,
            min_duration=self.min_duration,
            per_turn=self.per_turn,
            per_label=self.per_label,
            per_fold=self.per_fold,
            per_epoch=self.per_epoch,
            label_min_duration=self.label_min_duration,
        )

    @property
    def max_distance(self):
        if self.metric == "cosine":
            return 2.0
        elif self.metric == "angular":
            return np.pi
        elif self.metric == "euclidean":
            # FIXME. incorrect if embedding are not unit-normalized
            return 2.0
        else:
            msg = "'metric' must be one of {'euclidean', 'cosine', 'angular'}."
            raise ValueError(msg)

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.Tensor
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix
        """

        if self.metric == "euclidean":
            return F.pdist(fX)

        elif self.metric in ("cosine", "angular"):

            distance = 0.5 * torch.pow(F.pdist(F.normalize(fX)), 2)
            if self.metric == "cosine":
                return distance

            return torch.acos(torch.clamp(1.0 - distance, -1 + 1e-12, 1 - 1e-12))

    def embed(self, batch):
        """Extract embeddings (and aggregate per turn)

        Parameters
        ----------
        batch : `dict`
            ['X'] (batch_size, n_samples, n_features) `np.ndarray`
            ['y'] (batch_size, ) `np.ndarray`

        Returns
        -------
        fX : (batch_size / per_turn, n_dimensions) `torch.Tensor`
        y : (batch_size / per_turn, ) `np.ndarray`
        """

        X = torch.tensor(batch["X"], dtype=torch.float32, device=self.device_)
        fX = self.model_(X)

        if self.per_turn > 1:
            # TODO. add support for other aggregation functions, e.g. replacing
            # mean by product may encourage sparse representation
            agg_fX = fX.view(self.per_fold * self.per_label, self.per_turn, -1).mean(
                axis=1
            )

            agg_y = batch["y"][:: self.per_turn]

        else:
            agg_fX = fX
            agg_y = batch["y"]

        return agg_fX, agg_y

    def to_numpy(self, tensor):
        """Convert torch.Tensor to numpy array"""
        cpu = torch.device("cpu")
        return tensor.detach().to(cpu).numpy()

    @property
    def task(self):
        return Task(type=TaskType.REPRESENTATION_LEARNING, output=TaskOutput.VECTOR)
