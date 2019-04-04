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

import time
import numpy as np
from tqdm import tqdm
from .callback import Callback


class Logging(Callback):
    """Log loss and processing time to tensorboard and progress bar"""

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.losses_ = list()
        self.times_ = list()

    def on_train_start(self, trainer):
        self.epochs_pbar_ = tqdm(
            desc=f'Training',
            total=self.epochs, leave=True, ncols=80,
            unit='epoch', initial=trainer.epoch_, position=0)

    def on_epoch_start(self, trainer):
        self.epochs_pbar_.update(1)

        self.batches_pbar_ = tqdm(
            desc=f'Epoch #{trainer.epoch_}',
            total=trainer.batches_per_epoch_,
            leave=False, ncols=80,
            unit='batch', position=1)

    def on_batch_start(self, trainer, batch):
        self.time_ = time.time()

    def on_batch_end(self, trainer, loss):
        # TODO. use smoothed running mean instead...
        self.losses_.append(loss)
        self.times_.append(time.time() - self.time_)

        self.batches_pbar_.set_postfix(
            ordered_dict={'loss': np.mean(self.losses_)})
        self.batches_pbar_.update(1)

    def on_epoch_end(self, trainer):
        # TODO. use smoothed running mean instead...
        trainer.tensorboard_.add_scalar(
            'loss', np.mean(self.losses_),
            global_step=trainer.epoch_)
        self.losses_.clear()

        trainer.tensorboard_.add_scalar(
            'time/forward_backward', np.mean(self.times_),
            global_step=trainer.epoch_)
        self.times_.clear()
