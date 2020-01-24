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
# Juan Manuel CORIA

import torch
from torch import nn
import torch.nn.functional as F
from .classification import Classification


class CocoLinear(nn.Module):
    """Congenerous Cosine linear module (for CoCo loss)

        Parameters
        ----------
        nfeat : int
            Embedding dimension
        nclass : int
            Number of classes
        alpha : float
            Scaling factor used in embedding L2-normalization
        """

    def __init__(self, nfeat, nclass, alpha):
        super(CocoLinear, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))

    def forward(self, x):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        Returns
        -------
        fX : `torch.Tensor`
            logits after the congenerous cosine transformation
        """
        # normalize centers
        cnorm = F.normalize(self.centers)
        # normalize scaled embeddings
        xnorm = self.alpha * F.normalize(x)
        # calculate logits like in `nn.Linear`
        logits = torch.matmul(xnorm, torch.transpose(cnorm, 0, 1))
        return logits


class CongenerousCosineLoss(Classification):
    """Train embeddings as last hidden layer of an additive angular margin loss classifier

    Parameters
    ----------
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
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
    alpha : float
        Scaling factor used in embedding L2-normalization. Defaults to 6.25.
    """

    def __init__(self, duration=1.0,
                       per_label=1,
                       per_fold=32,
                       per_epoch: float = None,
                       label_min_duration=0.,
                       alpha=6.25):

        super().__init__(duration=duration,
                         per_label=per_label,
                         per_fold=per_fold,
                         per_epoch=per_epoch,
                         label_min_duration=label_min_duration)
        self.alpha = alpha

    def more_parameters(self):
        """Initialize trainable trainer parameters

        Yields
        ------
        parameter : nn.Parameter
            Trainable trainer parameters
        """

        self.classifier_ = CocoLinear(
            self.model.dimension,
            len(self.specifications['y']['classes']),
            self.alpha).to(self.device)

        return self.classifier_.parameters()
