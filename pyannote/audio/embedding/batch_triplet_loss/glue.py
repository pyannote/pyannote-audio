#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
# Grégory GELLY
# Hervé BREDIN - http://herve.niderb.fr

import numpy as np
from functools import partial

from pyannote.audio.embedding.glue import BatchGlue


def triplet_loss(inputs, distance=None):

    embeddings = inputs[0]
    labels = inputs[1]

    cost = 0.0

    d_embeddings = 0.0 * embeddings

    for ii, (anchor, anchor_label) in enumerate(zip(embeddings, labels)):
        for kk, (positive, positive_label) in enumerate(zip(embeddings, labels)):

            if (ii == kk) or (anchor_label != positive_label):
                continue

                for ll, (negative, negative_label) in enumerate(zip(embeddings, labels)):

                    if negative_label == positive_label:
                        continue

                    [cost_, d_anchor_, d_positive_, d_negative_] = distance(
                        anchor, positive, negative)
                    cost += cost_
                    d_embeddings[ii, :] += d_anchor_
                    d_embeddings[kk, :] += d_positive_
                    d_embeddings[ll, :] += d_negative_

    return [cost, d_embeddings]


class TripletLoss(BatchGlue):
    """Triplet loss for sequence embedding

            anchor        |-----------|           |---------|
            input    -->  | embedding | --> a --> |         |
            sequence      |-----------|           |         |
                                                  |         |
            positive      |-----------|           | triplet |
            input    -->  | embedding | --> p --> |         | --> loss value
            sequence      |-----------|           |  loss   |
                                                  |         |
            negative      |-----------|           |         |
            input    -->  | embedding | --> n --> |         |
            sequence      |-----------|           |---------|
    """

    def compute_derivatives(self, embeddings, labels):

        embeddings = embeddings.astype('float64')

        folds = []
        fold_size = self.per_fold * self.per_label

        for t in range(self.per_batch):
            fX = embeddings[t * fold_size:(t+1) * fold_size]
            y = labels[t * fold_size:(t+1) * fold_size]
            folds.append([fX, y, self.loss_])

        # self.loss_ is shared by all subsequent calls to 'triplet_loss'
        process_fold = partial(triplet_loss, distance=self.loss_)

        # TODO - use zip instead of this for loop
        costs = []
        derivatives = []
        for output in self.pool_.imap(process_fold, folds):
            costs.append(output[0])
            derivatives.append(output[1])

        return [np.hstack(costs), np.vstack(derivatives)]
