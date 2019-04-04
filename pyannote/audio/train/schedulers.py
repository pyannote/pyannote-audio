#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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


import numpy as np
from collections import deque
from dlib import count_steps_without_decrease
from dlib import count_steps_without_decrease_robust
from .callback import Callback


class ConstantScheduler(Callback):
    """Constant learning rate

    Parameters
    ----------
    max_lr : {float, list}, optional
        Initial learning rate. Defaults to using optimizer's own learning rate.

    Example
    -------
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> batches_per_epoch = 1000
    >>> scheduler = ConstantScheduler(optimizer, batches_per_epoch)
    """

    def __init__(self, max_lr=None, **kwargs):
        super().__init__()
        self.max_lr = max_lr

    def on_epoch_end(self, trainer):
        # TODO. save state to dict
        pass

    def load_epoch(self, trainer, epoch):
        # TODO. load state from dict
        pass

    def on_train_start(self, trainer):

        if self.max_lr is None:
            lrs = [g['lr'] for g in trainer.optimizer_.param_groups]

        elif isinstance(max_lr, (list, tuple)):
            lrs = max_lr

        else:
            lrs = [max_lr] * len(trainer.optimizer_.param_groups)

        for param_group, lr in zip(trainer.optimizer_.param_groups, lrs):
            param_group['lr'] = lr

        self.lr_ = lr

    def on_epoch_start(self, trainer):
        trainer.tensorboard_.add_scalar(
            f'lr', self.lr_, global_step=trainer.epoch_)

    def on_batch_end(self, trainer, loss):
        pass


class DavisKingScheduler(Callback):
    """Automatic Learning Rate Scheduling That Really Works

    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

    Parameters
    ----------
    max_lr : {float, list}, optional
        Initial learning rate. Defaults to using optimizer's own learning rate.
    factor : float, optional
        Factor by which the learning rate will be reduced.
        new_lr = old_lr * factor. Defaults to 0.3
    patience : int, optional
        Number of epochs with no improvement after which learning rate will
        be reduced. Defaults to 10.
    """

    def __init__(self, max_lr=None, factor=0.3, patience=10, **kwargs):
        super().__init__()
        self.max_lr = max_lr
        self.factor = factor
        self.patience = patience

    def on_epoch_end(self, trainer):
        # TODO. save state to dict
        pass

    def load_epoch(self, trainer, epoch):
        # TODO. load state from dict
        pass

    def on_train_start(self, trainer):

        # learning rate upper bound
        if self.max_lr is None:
            self.max_lrs_ = [g['lr'] for g in trainer.optimizer_.param_groups]

        elif isinstance(max_lr, (list, tuple)):
            self.max_lrs_ = [lr for lr in self.max_lr]

        else:
            self.max_lrs_ = [self.max_lr] * len(trainer.optimizer_.param_groups)

        # initialize optimizer learning rate
        for param_group, lr in zip(trainer.optimizer_.param_groups, self.max_lrs_):
            param_group['lr'] = lr

        # TODO check in dlib's code whether patience * batches_per_epoch + 1
        # would actually be enough
        maxlen = 10 * self.patience * trainer.batches_per_epoch_
        self.losses_ = deque([], maxlen=maxlen)

        self.lr_ = lr

    def on_epoch_start(self, trainer):
        trainer.tensorboard_.add_scalar(
            f'lr', self.lr_, global_step=trainer.epoch_)

    def on_batch_end(self, trainer, loss):

        # store current batch loss
        self.losses_.append(loss)

        # compute statistics on batch loss trend
        count = count_steps_without_decrease(self.losses_)
        count_robust = count_steps_without_decrease_robust(self.losses_)

        # if batch loss hasn't been decreasing for a while
        patience = self.patience * trainer.batches_per_epoch_
        if count > patience and count_robust > patience:
            factor = self.factor
            # reset batch loss trend
            self.losses_.clear()
        else:
            factor = 1.

        # decrease optimizer learning rate
        for param_group in trainer.optimizer_.param_groups:
            lr = param_group['lr'] * factor
            param_group['lr'] = lr

        self.lr_ = lr

class CyclicScheduler(Callback):
    """

    Parameters
    ----------
    min_lr, max_lr : {float, list}, optional
        Learning rate bounds.
    min_momentum, max_momentum : float, optional
        Momentum bounds.
    epochs_per_cycle : int, optional
        Number of epochs per cycle. Defaults to 20.
    decay : float, optional
        Defaults to 1 (i.e. no decay).
    """

    def __init__(self, min_lr=None, max_lr=None,
                 min_momentum=0.85, max_momentum=0.95, epochs_per_cycle=20,
                 decay=1., **kwargs):

        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.epochs_per_cycle = epochs_per_cycle
        self.decay = decay

    def on_epoch_end(self, trainer):
        # TODO. save state to dict
        pass

    def load_epoch(self, trainer, epoch):
        # TODO. load state from dict
        pass

    def on_train_start(self, trainer):

        # learning rate upper bound
        if self.max_lr is None:
            self.max_lrs_ = [g['lr'] for g in trainer.optimizer_.param_groups]

        elif isinstance(self.max_lr, (list, tuple)):
            self.max_lrs_ = [lr for lr in self.max_lr]

        else:
            self.max_lrs_ = [self.max_lr] * len(trainer.optimizer_.param_groups)

        # learning rate lower bound
        if self.min_lr is None:
            self.min_lrs_ = [0.1 * lr for lr in self.max_lrs_]

        elif isinstance(min_lr, (list, tuple)):
            self.min_lrs_ = [lr for lr in self.min_lr]

        else:
            self.min_lrs_ = [self.min_lr] * len(trainer.optimizer_.param_groups)

        # initialize optimizer learning rate to lower value
        # and momentum to higher value
        for param_group, lr in zip(trainer.optimizer_.param_groups,
                                   self.min_lrs_):
            param_group['lr'] = lr
            param_group['momentum'] = self.max_momentum

        self.n_batches_ = 0
        self.lr_ = lr

    def on_epoch_start(self, trainer):
        trainer.tensorboard_.add_scalar(
            f'lr', self.lr_, global_step=trainer.epoch_)

    def on_batch_end(self, trainer, loss):

        # bpc = batches per cycle
        bpc = self.epochs_per_cycle * trainer.batches_per_epoch_

        # current cycle (1 for first cycle, 2 for second cycle, etc.)
        cycle = np.floor(1 + .5 * self.n_batches_ / bpc)

        # position within current cycle
        rho = max(0, 1 - np.abs(self.n_batches_ / bpc - 2 * cycle + 1))

        # update learning rates and momentum
        momentum = self.max_momentum - \
            (self.max_momentum - self.min_momentum) * rho
        group_min_max = zip(trainer.optimizer_.param_groups,
                            self.min_lrs_,  self.max_lrs_)
        for param_group, min_lr, max_lr in group_min_max:

            # cycle decay
            max_lr = self.decay ** (cycle - 1) * max_lr
            lr = min_lr + (max_lr - min_lr) * rho
            param_group['lr'] = lr
            param_group['momentum'] = momentum

        self.n_batches_ += 1
        self.lr_ = lr
