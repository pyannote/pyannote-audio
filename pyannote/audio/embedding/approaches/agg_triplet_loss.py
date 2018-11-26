#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

import itertools
import numpy as np
import torch
import torch.nn.functional as F
from pyannote.audio.embedding.generators import SpeechTurnSubSegmentGenerator
from .triplet_loss import TripletLoss


class AggTripletLoss(TripletLoss):
    """

    Parameters
    ----------
    per_turn : int, optional
        Number of segments per speech turn. A heuristic may use a lower value
        to reduce the number of overlapping segments in case of short speech
        turns. Defaults to 2.
    rescale : {'softmax'}, optional
        Rescale embeddings before aggregation. Defaults to **not** rescale.

    """

    def __init__(self, metric='angular', margin=0.2, clamp='positive',
                 duration=3., sampling='all', parallel=1,
                 per_label=3, per_fold=None, per_turn=2,
                 rescale=None):

        super(AggTripletLoss, self).__init__(
            duration=duration, metric=metric, margin=margin, clamp=clamp,
            sampling=sampling, per_label=per_label, per_fold=per_fold,
            parallel=parallel)

        self.per_turn = per_turn
        self.rescale = rescale

    def aggregate(self, batch):

        fX_avg, y = [], []

        fX = batch['fX']
        n_segments = len(fX)

        # iterate over segments, speech turn by speech turn
        nyz = np.vstack([np.arange(n_segments), batch['y'], batch['z']]).T
        for _, nyz_ in itertools.groupby(nyz, lambda v: v[2]):

            # (n, 3) numpy array where
            # * n is the number of segments in current speech turn
            # * dim #0 is the index of segment in original batch
            # * dim #1 is the label of speech turn
            # * dim #2 is the index of speech turn (used for grouping)
            nyz_ = np.stack(nyz_)

            # compute (and stack) average embedding over all segments
            # of current speech turn
            indices = nyz_[:, 0]

            fX_ = fX[indices]
            if self.rescale == 'softmax':
                old_norm = torch.norm(fX_, 2, 1, keepdim=True)
                new_norm = F.softmax(old_norm, dim=0)
                rescaled = new_norm / old_norm * fX_

            else:
                rescaled = fX_

            fX_avg.append(torch.mean(rescaled, dim=0, keepdim=True))

            # stack label of current speech turn
            # (for later triplet sampling)
            y.append(nyz_[0, 1])

        fX_avg = torch.cat(fX_avg, dim=0)
        y = np.array(y)

        batch['fX'] = fX_avg
        batch['y'] = y

        return batch

    def get_batch_generator(self, feature_extraction):
        return SpeechTurnSubSegmentGenerator(
            feature_extraction, self.duration,
            per_label=self.per_label, per_fold=self.per_fold,
            per_turn=self.per_turn, parallel=self.parallel)
