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
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from .base import EmbeddingApproach


class ContrastiveLoss(EmbeddingApproach):
    """Train embeddings using triplet loss

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin multiplicative factor. Defaults to 0.2.
    size_average : `bool`, optional
        Divide total loss by batch size
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep them all).
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    """

    def __init__(self, duration=1.0,
                       metric='cosine',
                       margin=0.2,
                       size_average=True,
                       per_label=3,
                       per_fold=None,
                       per_epoch: float = None,
                       label_min_duration=0.):

        super().__init__()

        self.metric = metric
        self.margin = margin

        self.margin_ = self.margin * self.max_distance

        self.size_average = size_average
        self.per_fold = per_fold
        self.per_label = per_label
        self.per_epoch = per_epoch
        self.label_min_duration = label_min_duration
        self.duration = duration

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
            ['loss'] (`torch.Tensor`) : Triplet loss
        """

        fX = self.forward(batch)
        y = batch['y']

        # calculate the distances between every sample in the batch
        nbatch = fX.size(0)
        dist = self.pdist(fX).to(self.device_)

        # calculate the ground truth for each pair
        gt = []
        for i in range(nbatch - 1):
            for j in range(i + 1, nbatch):
                gt.append(int(y[i] != y[j]))
        gt = torch.Tensor(gt).float().to(self.device_)

        # Calculate the losses as described in the paper
        losses = (1 - gt) * torch.pow(dist, 2) + gt * torch.pow(torch.clamp(self.margin_ - dist, min=1e-8), 2)
        losses = torch.sum(losses) / 2
        # Average by batch size if requested
        losses = losses / dist.size(0) if self.size_average else losses

        return {'loss': losses}
