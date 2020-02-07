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

import warnings
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from pyannote.audio.train.trainer import Trainer
import numpy as np


class EmbeddingApproach(Trainer):

    @property
    def max_distance(self):
        if self.metric == 'cosine':
            return 2.
        elif self.metric == 'angular':
            return np.pi
        elif self.metric == 'euclidean':
            # FIXME. incorrect if embedding are not unit-normalized
            return 2.
        else:
            msg = "'metric' must be one of {'euclidean', 'cosine', 'angular'}."
            raise ValueError(msg)

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.Tensor
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.Tensor
            Condensed pairwise distance matrix
        """

        if self.metric == 'euclidean':
            return F.pdist(fX)

        elif self.metric in ('cosine', 'angular'):

            distance = 0.5 * torch.pow(F.pdist(F.normalize(fX)), 2)
            if self.metric == 'cosine':
                return distance

            return torch.acos(torch.clamp(1. - distance,
                                          -1 + 1e-12,
                                          1 - 1e-12))

    def forward(self, batch):
        """Forward pass on current batch

        Parameters
        ----------
        batch : `dict`
            ['X'] (`list`of `numpy.ndarray`)

        Returns
        -------
        fX : `torch.Tensor`
            self.model_(batch['X'])
        """

        lengths = [len(x) for x in batch['X']]
        variable_lengths = len(set(lengths)) > 1

        # if sequences have variable lengths
        if variable_lengths:

            # TODO: use new pytorch feature that handle sorting automatically

            # sort them in order of length
            _, sort = torch.sort(torch.tensor(lengths), descending=True)
            _, unsort = torch.sort(sort)
            sequences = [torch.tensor(batch['X'][i],
                                      dtype=torch.float32,
                                      device=self.device_) for i in sort]

            # pack them if model supports PackedSequences
            if getattr(self.model_, 'supports_packed', False):
                batch['X'] = pack_sequence(sequences)
                fX = self.model_(batch['X'])

            # process them separately if model does not support PackedSequence
            else:
                try:
                    fX = torch.cat([self.model_(x.unsqueeze(0))
                                    for x in sequences])
                    msg = (
                        'Model does not support variable lengths batch, '
                        'so we are processing sequences separately...'
                    )
                    warnings.warn(msg)

                except ValueError as e:
                    min_length = min(lengths)
                    fX = self.model_(torch.stack([x[:min_length]
                                                  for x in sequences]))
                    msg = (
                        'Model does not support variable lengths batch, '
                        'so we cropped them all to the shortest one.'
                    )
                    warnings.warn(msg)

            return fX[unsort]

        # if sequences share the same length
        batch['X'] = torch.tensor(np.stack(batch['X']),
                                  dtype=torch.float32,
                                  device=self.device_)
        return self.model_(batch['X'])

    def to_numpy(self, tensor):
        """Convert torch.Tensor to numpy array"""
        cpu = torch.device('cpu')
        return tensor.detach().to(cpu).numpy()
