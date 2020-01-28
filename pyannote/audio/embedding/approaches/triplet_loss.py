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

from typing import Optional
import numpy as np
import torch
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.core.utils.distance import to_condensed
from scipy.spatial.distance import squareform
from pyannote.metrics.binary_classification import det_curve
from .base import EmbeddingApproach


class TripletLoss(EmbeddingApproach):
    """Train embeddings using triplet loss

    Parameters
    ----------
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin multiplicative factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard', 'negative', 'easy'}, optional
        Triplet sampling strategy. Defaults to 'all' (i.e. use all possible
        triplets). 'hard' sampling use both hardest positive and negative for
        each anchor. 'negative' sampling use hardest negative for each
        (anchor, positive) pairs. 'easy' sampling only use easy triplets (i.e.
        those for which d(anchor, positive) < d(anchor, negative)).
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    per_turn : int, optional
        Number of chunks per speech turn. Defaults to 1.
        If per_turn is greater than one, embeddings of the same speech turn
        are averaged before building triplets. The intuition is that it might
        help learn embeddings meant to be averaged/summed.
    per_label : int, optional
        Number of speech turns per speaker in each batch.
        Defaults to 3.
    per_fold : int, optional
        Number of different speakers in each batch.
        Defaults to all speakers.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep it all).

    Notes
    -----
    delta = d(anchor, positive) - d(anchor, negative)

    * with 'positive' clamping:
        loss = max(0, delta + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * delta)

    where d(., .) varies in range [0, D] (e.g. D=2 for euclidean distance).

    """

    def __init__(self, metric='cosine',
                       margin: float = 0.2,
                       clamp='positive',
                       sampling='all',
                       duration: float = 1.,
                       per_turn: int = 1,
                       per_label: int = 3,
                       per_fold: Optional[int] = None,
                       per_epoch: float = None,
                       label_min_duration: float = 0.):

        super().__init__()

        self.metric = metric
        self.margin = margin

        self.margin_ = self.margin * self.max_distance

        if clamp not in {'positive', 'sigmoid', 'softmargin'}:
            msg = "'clamp' must be one of {'positive', 'sigmoid', 'softmargin'}."
            raise ValueError(msg)
        self.clamp = clamp

        if sampling not in {'all', 'hard', 'negative', 'easy'}:
            msg = "'sampling' must be one of {'all', 'hard', 'negative', 'easy'}."
            raise ValueError(msg)
        self.sampling = sampling

        self.per_turn = per_turn
        self.per_fold = per_fold
        self.per_label = per_label
        self.per_epoch = per_epoch
        self.label_min_duration = label_min_duration

        self.duration = duration

    def batch_easy(self, y, distances):
        """Build easy triplets"""

        anchors, positives, negatives = [], [], []

        distances = squareform(self.to_numpy(distances))

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                d = distances[anchor, positive]

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    if d > distances[anchor, negative]:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives


    def batch_hard(self, y, distances):
        """Build triplet with both hardest positive and hardest negative

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        distances = squareform(self.to_numpy(distances))
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            d = distances[anchor]

            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

    def batch_negative(self, y, distances):
        """Build triplet with hardest negative

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        distances = squareform(self.to_numpy(distances))
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        return anchors, positives, negatives

    def batch_all(self, y, distances):
        """Build all possible triplet

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def triplet_loss(self, distances, anchors, positives, negatives,
                     return_delta=False):
        """Compute triplet loss

        Parameters
        ----------
        distances : torch.Tensor
            Condensed matrix of pairwise distances.
        anchors, positives, negatives : list of int
            Triplets indices.
        return_delta : bool, optional
            Return delta before clamping.

        Returns
        -------
        loss : torch.Tensor
            Triplet loss.
        """

        # estimate total number of embeddings from pdist shape
        n = int(.5 * (1 + np.sqrt(1 + 8 * len(distances))))

        # convert indices from squared matrix
        # to condensed matrix referential
        pos = to_condensed(n, anchors, positives)
        neg = to_condensed(n, anchors, negatives)

        # compute raw triplet loss (no margin, no clamping)
        # the lower, the better
        delta = distances[pos] - distances[neg]

        # clamp triplet loss
        if self.clamp == 'positive':
            loss = torch.clamp(delta + self.margin_, min=0)

        elif self.clamp == 'softmargin':
            loss = torch.log1p(torch.exp(delta))

        elif self.clamp == 'sigmoid':
            # TODO. tune this "10" hyperparameter
            # TODO. log-sigmoid
            loss = torch.sigmoid(10 * (delta + self.margin_))

        # return triplet losses
        if return_delta:
            return loss, delta.view((-1, 1)), pos, neg
        else:
            return loss

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
            duration=self.duration,
            per_turn=self.per_turn,
            per_label=self.per_label,
            per_fold=self.per_fold,
            per_epoch=self.per_epoch,
            label_min_duration=self.label_min_duration)

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
            ['loss'] (`torch.Tensor`) : Triplet loss
        """

        fX = self.forward(batch)

        # If per_turn is greater than one, embeddings of the same speech turn
        # are averaged before building triplets. The intuition is that it might
        # help learn embeddings meant to be averaged/summed.
        if self.per_turn > 1:
            y = batch['y'][::self.per_turn]
            avg_fX = fX.view(self.per_fold * self.per_label,
                             self.per_turn, -1).mean(axis=1)
            distances = self.pdist(avg_fX)

        else:
            y = batch['y']
            distances = self.pdist(fX)

        # sample triplets
        triplets = getattr(self, 'batch_{0}'.format(self.sampling))
        anchors, positives, negatives = triplets(y, distances)

        # compute loss for each triplet
        losses, deltas, _, _ = self.triplet_loss(
            distances, anchors, positives, negatives,
            return_delta=True)

        # average over all triplets
        return {'loss': torch.mean(losses)}
