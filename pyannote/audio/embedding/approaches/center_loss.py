#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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
# Juan Manuel CORIA

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from .base import EmbeddingApproach


class CenterLoss(EmbeddingApproach):
    """Train embeddings as last hidden layer of a classifier

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    loss_weight : float, optional
        Lambda parameter controlling the effect of center distance in the loss value.
        Defaults to 1.
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

    CLASSIFIER_PT = '{train_dir}/weights/{epoch:04d}.classifier.pt'
    CENTERS_PT = '{train_dir}/weights/{epoch:04d}.centers.pt'

    def __init__(self, duration=1.0,
                       per_label=1,
                       per_fold=32,
                       loss_weight=1.,
                       per_epoch: float = None,
                       label_min_duration=0.):
        super().__init__()

        self.loss_weight = loss_weight
        self.per_label = per_label
        self.per_fold = per_fold
        self.per_epoch = per_epoch
        self.label_min_duration = label_min_duration

        self.duration = duration

        self.logsoftmax_ = nn.LogSoftmax(dim=1)
        self.nll = nn.NLLLoss()

    def more_parameters(self):
        """Initialize trainable trainer parameters

        Yields
        ------
        parameter : nn.Parameter
            Trainable trainer parameters
        """

        n_classes = len(self.specifications['y']['classes'])
        self.classifier_ = nn.Linear(self.model.dimension,
                                     n_classes, bias=False).to(self.device)
        self.center_dist_ = CenterDistanceModule(self.model.dimension,
                                                 n_classes).to(self.device)

        return chain(self.classifier_.parameters(),
                     self.center_dist_.parameters())

    def load_more(self, model_pt=None):
        """Load classifier and centers from disk"""

        if model_pt is None:
            classifier_pt = self.CLASSIFIER_PT.format(
                train_dir=self.train_dir_, epoch=self.epoch_)
            centers_pt = self.CENTERS_PT.format(
                train_dir=self.train_dir_, epoch=self.epoch_)

        else:
            classifier_pt = model_pt.with_suffix('.classifier.pt')
            centers_pt = model_pt.with_suffix('.centers.pt')

        classifier_state = torch.load(
            classifier_pt, map_location=lambda storage, loc: storage)
        self.classifier_.load_state_dict(classifier_state)

        centers_state = torch.load(
            centers_pt, map_location=lambda storage, loc: storage)
        self.center_dist_.load_state_dict(centers_state)

    def save_more(self):
        """Save classifier and centers to disk"""

        classifier_pt = self.CLASSIFIER_PT.format(
            train_dir=self.train_dir_, epoch=self.epoch_)
        centers_pt = self.CENTERS_PT.format(
            train_dir=self.train_dir_, epoch=self.epoch_)

        torch.save(self.classifier_.state_dict(), classifier_pt)
        torch.save(self.center_dist_.state_dict(), centers_pt)

    def get_batch_generator(self, feature_extraction,
                            protocol, subset='train',
                            **kwargs):
        """Get batch generator

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`
        protocol : `pyannote.database.Protocol`
        subset : {'train', 'development', 'test'}, optional

        Returns
        -------
        generator : `pyannote.audio.embedding.generators.SpeechSegmentGenerator`
        """

        return SpeechSegmentGenerator(
            feature_extraction,
            protocol,
            subset=subset,
            label_min_duration=self.label_min_duration,
            per_label=self.per_label,
            per_fold=self.per_fold,
            per_epoch=self.per_epoch,
            duration=self.duration)

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

        # extract embeddings
        fX = self.forward(batch)

        # apply classification layer
        scores = self.logsoftmax_(self.classifier_(fX))

        # get labels as a tensor
        target = torch.tensor(
            batch['y'],
            dtype=torch.int64,
            device=self.device_)

        # compute center loss
        return {'loss': self.nll(scores, target) + self.loss_weight * self.center_dist_(fX, target)}


class CenterDistanceModule(nn.Module):
    """Sum of embedding-to-center cosine distances

    Parameters
    ----------
    nfeat : int
        Embedding size
    nclass : int
        Number of classes
    """

    # TODO add support for other metrics (ex. euclidean, angular)

    def __init__(self, nfeat, nclass):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))
        self.nfeat = nfeat

    def forward(self, feat, y):
        """Calculate the sum of cosine distances from embeddings to centers

        Parameters
        ----------
        feat : `torch.Tensor`
            Embedding batch
        y : `torch.Tensor`
            Non one-hot labels
        Returns
        -------
        distance_sum : float
            Sum of cosine distances from embeddings to centers
        """
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # Select appropriate centers for this batch's labels
        centers_batch = self.centers.index_select(0, y.long())
        # Return the sum of the squared distance normalized by the batch size
        dists = 1 - F.cosine_similarity(feat, centers_batch, dim=1, eps=1e-8)
        return torch.sum(torch.pow(dists, 2)) / 2.0 / batch_size
