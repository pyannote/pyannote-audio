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

import warnings
import torch
import torch.nn as nn
from .base import RepresentationLearning


class Linear(nn.Linear):
    def forward(self, x, target=None):
        return super().forward(x)


class Classification(RepresentationLearning):
    """Classification

    TODO explain

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
    bias : `bool`, optional
        Use bias in the classification layer
        Defaults to False.
    """

    # TODO. add option to see this classification step
    #       as cosine similarity to centroids (ie center loss?)

    CLASSIFIER_PT = "{train_dir}/weights/{epoch:04d}.classifier.pt"

    def __init__(
        self,
        duration: float = 1.0,
        min_duration: float = None,
        per_turn: int = 1,
        per_label: int = 1,
        per_fold: int = 32,
        per_epoch: float = None,
        label_min_duration: float = 0.0,
        bias: bool = False,
    ):

        super().__init__(
            duration=duration,
            min_duration=min_duration,
            per_turn=per_turn,
            per_label=per_label,
            per_fold=per_fold,
            per_epoch=per_epoch,
            label_min_duration=label_min_duration,
        )

        self.bias = bias
        self.logsoftmax_ = nn.LogSoftmax(dim=1)
        self.loss_ = nn.NLLLoss()

    def more_parameters(self):
        """Initialize trainable trainer parameters

        Returns
        -------
        parameters : iterable
            Trainable trainer parameters
        """

        self.classifier_ = Linear(
            self.model.dimension,
            len(self.specifications["y"]["classes"]),
            bias=self.bias,
        ).to(self.device)

        return self.classifier_.parameters()

    def load_more(self, model_pt=None) -> bool:
        """Load classifier from disk

        Returns
        -------
        success : bool
            True if state was loaded successfully, False otherwise.
        """

        if model_pt is None:
            classifier_pt = self.CLASSIFIER_PT.format(
                train_dir=self.train_dir_, epoch=self.epoch_
            )
        else:
            classifier_pt = model_pt.with_suffix(".classifier.pt")

        try:
            classifier_state = torch.load(
                classifier_pt, map_location=lambda storage, loc: storage
            )
            self.classifier_.load_state_dict(classifier_state)
            success = True
        except Exception as e:
            msg = (
                f"Did not load classifier state (most likely because current "
                f"training session uses a different training set than the one "
                f"used for pre-training)."
            )
            warnings.warn(msg)
            success = False

        return success

    def save_more(self):
        """Save classifier weights to disk"""

        classifier_pt = self.CLASSIFIER_PT.format(
            train_dir=self.train_dir_, epoch=self.epoch_
        )
        torch.save(self.classifier_.state_dict(), classifier_pt)

    @property
    def metric(self):
        return "cosine"

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Cross-entropy loss
        """

        # extract and aggregate embeddings
        fX, y = self.embed(batch)
        target = torch.tensor(y, dtype=torch.int64, device=self.device)

        # apply classification layer
        logits = self.logsoftmax_(self.classifier_(fX, target=target))

        # compute classification loss
        loss_classification = self.loss_(logits, target)

        return {
            "loss": loss_classification,
            # add this for Tensorboard comparison with other compound losses
            "loss_classification": loss_classification,
        }
