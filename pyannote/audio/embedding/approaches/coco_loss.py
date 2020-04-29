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
# Juan Manuel CORIA - https://juanmc2005.github.io

import math
import warnings
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
    scale : float
        Scaling factor used in embedding L2-normalization
    """

    def __init__(self, nfeat, nclass, scale):
        super(CocoLinear, self).__init__()
        self.scale = scale
        self.centers = nn.Parameter(torch.randn(nclass, nfeat))

    def forward(self, x, target=None):
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
        xnorm = self.scale * F.normalize(x)
        # calculate logits like in `nn.Linear`
        logits = torch.matmul(xnorm, torch.transpose(cnorm, 0, 1))
        return logits


class CongenerousCosineLoss(Classification):
    """Congenerous cosine loss

    Train embeddings by maximizing the cosine similarity between
    an embedding and its class center.
    A hyper-parameter `alpha` is used to scale logits before
    applying softmax.

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
    scale : float
        Scaling factor used in embedding L2-normalization. Defaults to sqrt(2) * log(n_classes - 1).

    Reference
    ---------
    Rethinking Feature Discrimination and Polymerization for Large-scale Recognition
    https://arxiv.org/abs/1710.00870
    """

    def __init__(
        self,
        duration: float = 1.0,
        min_duration: float = None,
        per_turn: int = 1,
        per_label: int = 1,
        per_fold: int = 32,
        per_epoch: float = None,
        label_min_duration: float = 0.0,
        # `alpha` is deprecated in favor of `scale`
        alpha: float = None,
        scale: float = None
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

        if alpha is not None:
            msg = "The 'alpha' parameter is deprecated in favor of 'scale', " \
                  "and will be removed in a future release"
            warnings.warn(msg, FutureWarning)
            self.scale = alpha
        else:
            self.scale = scale

    def more_parameters(self):
        """Initialize trainable trainer parameters

        Yields
        ------
        parameter : nn.Parameter
            Trainable trainer parameters
        """

        nclass = len(self.specifications["y"]["classes"])
        # Use scaling initialization trick from AdaCos
        # Reference: https://arxiv.org/abs/1905.00292
        scale = math.sqrt(2) * math.log(nclass - 1) if self.scale is None else self.scale

        self.classifier_ = CocoLinear(
            self.model.dimension, nclass, scale
        ).to(self.device)

        return self.classifier_.parameters()
