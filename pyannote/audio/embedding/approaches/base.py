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


import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from pyannote.audio.train.trainer import Trainer
import numpy as np


class EmbeddingApproach(Trainer):

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

        if variable_lengths:
            _, sort = torch.sort(torch.tensor(lengths), descending=True)
            _, unsort = torch.sort(sort)
            sequences = [torch.tensor(batch['X'][i],
                                      dtype=torch.float32,
                                      device=self.device_) for i in sort]
            batch['X'] = pack_sequence(sequences)
        else:
            batch['X'] = torch.tensor(np.stack(batch['X']),
                                      dtype=torch.float32,
                                      device=self.device_)

        # forward pass
        fX = self.model_(batch['X'])

        # TODO. add support for structured fX
        if variable_lengths:
            fX = fX[unsort]

        return fX

    def to_numpy(self, tensor):
        """Convert torch.Tensor to numpy array"""
        cpu = torch.device('cpu')
        return tensor.detach().to(cpu).numpy()
