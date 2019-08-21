#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
from .classification import Classification


class ArcLinear(nn.Module):
    """Additive Angular Margin linear module (ArcFace)

    Parameters
    ----------
    nfeat : int
        Embedding dimension
    nclass : int
        Number of classes
    margin : float
        Angular margin to penalize distances between embeddings and centers
    s : float
        Scaling factor for the logits
    """

    def __init__(self, nfeat, nclass, margin, s):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = nn.Parameter(torch.Tensor(nclass, nfeat))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        y : `torch.Tensor`
            a non one-hot label batch
        Returns
        -------
        fX : `torch.Tensor`
            logits after the angular margin transformation
        """
        # normalize the feature vectors and W
        xnorm = F.normalize(x)
        Wnorm = F.normalize(self.W)
        y = y.long().view(-1, 1)
        # calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, y)
        # for numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # one hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, y, 1.0)
        # project margin differences into cosθj
        cos_theta_j += one_hot * (cos_theta_yi_margin - cos_theta_yi)
        # apply the scaling
        cos_theta_j = self.s * cos_theta_j
        return cos_theta_j


class AdditiveAngularMarginLoss(Classification):
    """Train embeddings as last hidden layer of an additive angular margin loss classifier

        Parameters
        ----------
        duration : float, optional
            Use fixed duration segments with this `duration`.
            Defaults (None) to using variable duration segments.
        min_duration : float, optional
            In case `duration` is None, set segment minimum duration.
        max_duration : float, optional
            In case `duration` is None, set segment maximum duration.
        per_label : `int`, optional
            Number of sequences per speaker in each batch. Defaults to 1.
        per_fold : `int`, optional
            Number of different speakers per batch. Defaults to 32.
        per_epoch : `float`, optional
            Number of days per epoch. Defaults to 7 (a week).
        label_min_duration : `float`, optional
            Remove speakers with less than that many seconds of speech.
            Defaults to 0 (i.e. keep them all).
        parallel : int, optional
            Number of prefetching background generators. Defaults to 1.
            Each generator will prefetch enough batches to cover a whole epoch.
            Set `parallel` to 0 to not use background generators.
        margin : float, optional
            Angular margin value. Defaults to 0.1.
        s : float, optional
            Scaling parameter value for the logits. Defaults to 7.
        """

    def __init__(self, duration=None, min_duration=None, max_duration=None,
                 per_label=1, per_fold=32, per_epoch=7, parallel=1,
                 label_min_duration=0., margin=0.1, s=7.0):
        super().__init__(duration=duration, min_duration=min_duration, max_duration=max_duration,
                         per_label=per_label, per_fold=per_fold, per_epoch=per_epoch, parallel=parallel,
                         label_min_duration=label_min_duration)
        self.margin = margin
        self.s = s

    def parameters(self, model, specifications, device):
        """Initialize trainable trainer parameters

        Parameters
        ----------
        specifications : `dict`
            Batch specs.

        Returns
        -------
        parameters : iterable
            Trainable trainer parameters
        """

        self.classifier_ = ArcLinear(
            model.dimension,
            len(specifications['y']['classes']),
            self.margin,
            self.s).to(device)

        return list(self.classifier_.parameters())

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
            ['loss'] (`torch.Tensor`) : Additive angular margin loss
        """

        # extract embeddings
        fX = self.forward(batch)

        # transform labels into tensor
        target = torch.tensor(
            batch['y'],
            dtype=torch.int64,
            device=self.device_)

        # apply classification layer
        scores = self.logsoftmax_(self.classifier_(fX, target))

        # compute classification loss
        return {'loss': self.loss_(scores, target)}
