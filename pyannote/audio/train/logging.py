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
    """Log loss and processing time to tensorboard and progress bar

    Parameters
    ----------
    epochs : `int`, optional
        Total number of epochs. Main progress bar will be prettier if provided.
    verbosity : `int`, optional
        Set level of verbosity.
        Use 0 (default) to not show any progress bar.
        Use 1 to show a progress bar updated at the end of each epoch.
        Use 2 to add a second progress bar updated at the end of each batch.
    """

    def __init__(self, epochs: int = None, verbosity: int = 0):
        super().__init__()
        self.epochs = epochs
        self.verbosity = verbosity
        self.beta_ = 0.98

    def on_train_start(self, trainer):
        if self.verbosity > 0:
            self.epochs_pbar_ = tqdm(
                desc=f"Training",
                total=self.epochs,
                leave=True,
                ncols=80,
                unit="epoch",
                initial=trainer.epoch_,
                position=0,
            )

    def on_epoch_start(self, trainer):

        # time spent in batch generation
        self.t_batch_ = list()
        # time spent in forward/backward
        self.t_model_ = list()

        # loss moving average
        self.n_batches_ = 0
        self.loss_moving_avg_ = dict()

        if self.verbosity > 0:
            self.epochs_pbar_.update(1)

        if self.verbosity > 1:
            self.batches_pbar_ = tqdm(
                desc=f"Epoch #{trainer.epoch_}",
                total=trainer.batches_per_epoch_,
                leave=False,
                ncols=80,
                unit="batch",
                position=1,
            )

        self.t_batch_end_ = time.time()

    def on_batch_start(self, trainer, batch):

        # mark time just before forward/backward
        self.t_batch_start_ = time.time()

        # time spent in batch generation
        self.t_batch_.append(self.t_batch_start_ - self.t_batch_end_)

        return batch

    def on_batch_end(self, trainer, batch_loss):

        # mark time just after forward/backward
        self.t_batch_end_ = time.time()

        # time spent in forward/backward
        self.t_model_.append(self.t_batch_end_ - self.t_batch_start_)

        self.n_batches_ += 1

        self.loss = dict()
        for key in batch_loss:
            if not key.startswith("loss"):
                continue
            loss = batch_loss[key].detach().cpu().item()
            self.loss_moving_avg_[key] = (
                self.beta_ * self.loss_moving_avg_.setdefault(key, 0.0)
                + (1 - self.beta_) * loss
            )
            self.loss[key] = self.loss_moving_avg_[key] / (
                1 - self.beta_ ** self.n_batches_
            )

        if self.verbosity > 1:
            self.batches_pbar_.set_postfix(ordered_dict=self.loss)
            self.batches_pbar_.update(1)

    def on_epoch_end(self, trainer):

        for key, loss in self.loss.items():
            trainer.tensorboard_.add_scalar(
                f"train/{key}", loss, global_step=trainer.epoch_
            )

        trainer.tensorboard_.add_histogram(
            "profiling/model",
            np.array(self.t_model_),
            global_step=trainer.epoch_,
            bins="fd",
        )

        trainer.tensorboard_.add_histogram(
            "profiling/batch",
            np.array(self.t_batch_),
            global_step=trainer.epoch_,
            bins="fd",
        )
