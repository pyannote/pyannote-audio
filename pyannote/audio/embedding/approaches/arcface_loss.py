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
import torch.nn as nn
import torch.nn.functional as F
from .classification import Classification


class ArcLinear(nn.Module):
    """Additive Angular Margin classification module

    Parameters
    ----------
    nfeat : int
        Embedding dimension
    nclass : int
        Number of classes
    margin : float
        Angular margin to penalize distances between embeddings and centers
    scale : float
        Scaling factor for the logits
    """

    def __init__(self, nfeat, nclass, margin, scale):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.scale = scale
        self.W = nn.Parameter(torch.Tensor(nclass, nfeat))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, target=None):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        target : `torch.Tensor`
            a non one-hot label batch

        Returns
        -------
        fX : `torch.Tensor`
            logits after the angular margin transformation
        """
        # normalize the feature vectors and W
        xnorm = F.normalize(x)
        Wnorm = F.normalize(self.W)
        target = target.long().view(-1, 1)
        # calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, target)
        # for numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # one hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, target, 1.0)
        # project margin differences into cosθj
        return self.scale * (cos_theta_j + one_hot * (cos_theta_yi_margin - cos_theta_yi))


class AdditiveAngularMarginLoss(Classification):
    """Additive angular margin loss

    Penalize the distance between an example and its class center
    with an angular margin, applying a scaling parameter to all logits.
    The intuition is to bring examples from a single class closer together
    in the embedding space by means of the angular margin.

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
    margin : float, optional
        Angular margin value. Defaults to 0.1.
    scale : float, optional
        Scaling parameter value for the logits. Defaults to sqrt(2) * log(n_classes - 1).

    Reference
    ---------
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    http://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
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
        margin: float = 0.1,
        # `s` is deprecated in favor of `scale`
        s: float = None,
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

        self.margin = margin
        if s is not None:
            msg = "The 's' parameter is deprecated in favor of 'scale', " \
                  "and will be removed in a future release"
            warnings.warn(msg, FutureWarning)
            self.scale = s
        else:
            self.scale = scale

    def more_parameters(self):
        """Initialize classifier layer

        Yields
        ------
        parameter : nn.Parameter
            Parameters
        """

        nclass = len(self.specifications["y"]["classes"])
        # Use scaling initialization trick from AdaCos
        # Reference: https://arxiv.org/abs/1905.00292
        scale = math.sqrt(2) * math.log(nclass - 1) if self.scale is None else self.scale

        self.classifier_ = ArcLinear(
            self.model.dimension,
            nclass,
            self.margin,
            scale,
        ).to(self.device)

        return self.classifier_.parameters()
