#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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


import os
import os.path
import torch
from pyannote.audio.util import mkdir_p


class Checkpoint(object):
    """

    Parameters
    ----------
    log_dir : str
    restart : boolean, optional
        Indicates that this training is a restart, not a cold start (default).
    """

    WEIGHTS_DIR = '{log_dir}/weights'
    WEIGHTS_PT = '{log_dir}/weights/{epoch:04d}.pt'
    OPTIMIZER_PT = '{log_dir}/weights/{epoch:04d}.optimizer.pt'

    def __init__(self, log_dir, restart=False):
        super(Checkpoint, self).__init__()

        # make sure path is absolute
        self.log_dir = os.path.realpath(log_dir)

        # create log_dir directory
        mkdir_p(self.log_dir)

        # this will fail if the directory already exists
        # and this is OK  because 'weights' directory
        # usually contains the output of very long computations
        # and you do not want to erase them by mistake :/
        self.restart = restart
        if not self.restart:
            weights_dir = self.WEIGHTS_DIR.format(log_dir=self.log_dir)
            os.makedirs(weights_dir)

    def on_epoch_end(self, epoch, model, optimizer):

        weights_pt = self.WEIGHTS_PT.format(
            log_dir=self.log_dir, epoch=epoch)
        torch.save(model.state_dict(), weights_pt)

        optimizer_pt = self.OPTIMIZER_PT.format(
            log_dir=self.log_dir, epoch=epoch)
        torch.save(optimizer.state_dict(), optimizer_pt)
