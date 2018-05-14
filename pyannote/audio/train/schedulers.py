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


import numpy as np
from collections import deque
from dlib import count_steps_without_decrease
from dlib import count_steps_without_decrease_robust


class DavisKingScheduler(object):
    """Automatic Learning Rate Scheduling That Really Works

    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    batches_per_epoch : int
        Number of batches per epoch.
    learning_rate : {float, list}, optional
        Initial learning rate. Defaults to using optimizer's own learning rate.
    factor : float, optional
        Factor by which the learning rate will be reduced.
        new_lr = old_lr * factor. Defaults to 0.9
    patience : int, optional
        Number of epochs with no improvement after which learning rate will
        be reduced. Defaults to 10.

    Example
    -------
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> batches_per_epoch = 1000
    >>> scheduler = DavisKingScheduler(optimizer, batches_per_epoch)
    >>> for mini_batch in batches:
    ...     mini_loss = train(mini_batch, optimizer)
    ...     scheduler.step(mini_loss)
    """

    def __init__(self, optimizer, batches_per_epoch,
                 learning_rate=None, factor=0.9, patience=10):

        super(DavisKingScheduler, self).__init__()
        self.batches_per_epoch = batches_per_epoch

        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # initialize optimizer learning rate
        if learning_rate is None:
            lrs = [g['lr'] for g in self.optimizer.param_groups]
        elif isinstance(learning_rate, (list, tuple)):
            lrs = learning_rate
        else:
            lrs = [learning_rate] * len(self.optimizer.param_groups)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        self.factor = factor
        self.patience = patience

        # TODO check in dlib's code whether patience * batches_per_epoch + 1
        # would actually be enough
        maxlen = 10 * self.patience * self.batches_per_epoch
        self.batch_losses_ = deque([], maxlen=maxlen)

    def batch_step(self, batch_loss):

        # store current batch loss
        self.batch_losses_.append(batch_loss)

        # compute statistics on batch loss trend
        count = count_steps_without_decrease(self.batch_losses_)
        count_robust = count_steps_without_decrease_robust(self.batch_losses_)

        # if batch loss hasn't been decreasing for a while
        patience = self.patience * self.batches_per_epoch
        if count > patience and count_robust > patience:

            # decrease optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor

            # reset batch loss trend
            self.batch_losses_.clear()

        return {
            'epochs_without_decrease': count / self.batches_per_epoch,
            'epochs_without_decrease_robust': \
                count_robust / self.batches_per_epoch}


class OneCycle(object):
    """

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    batches_per_epoch : int
        Number of batches per epoch.
    learning_rate : {float, list}, optional
        Learning rate upper bound. Defaults to using optimizer's own learning
        rate.
    epochs_per_cycle : int, optional
        Number of epochs per cycle.
    """

    def __init__(self, optimizer, batches_per_epoch, learning_rate=None,
                 epochs_per_cycle=20):

        super(OneCycle, self).__init__()
        self.batches_per_epoch = batches_per_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs_per_cycle = epochs_per_cycle

        if learning_rate is None:
            self.max_lrs_ = [g['lr'] for g in self.optimizer.param_groups]

        elif isinstance(learning_rate, (list, tuple)):
            self.max_lrs_ = [lr for lr in learning_rate]

        else:
            self.max_lrs_ = [learning_rate] * len(self.optimizer.param_groups)

        # TODO. make 0.1 tunable
        self.min_lrs_ = [0.1 * lr for lr in self.max_lrs_]

        # initialize omptimizer learning rate to lower value
        for param_group, lr in zip(self.optimizer.param_groups, self.min_lrs_):
            param_group['lr'] = lr

        self.n_batches_ = 0

    def batch_step(self, batch_loss):

        # bpc = batches per cycle
        bpc = self.epochs_per_cycle * self.batches_per_epoch

        # current cycle (1 for first cycle, 2 for second cycle, etc.)
        cycle = np.floor(1 + .5 * self.n_batches_ / bpc)

        # position within current cycle
        rho = max(0, 1 - np.abs(self.n_batches_ / bpc - 2 * cycle + 1))

        # update learning rates
        group_min_max = zip(self.optimizer.param_groups,
                            self.min_lrs_,  self.max_lrs_)
        for param_group, min_lr, max_lr in group_min_max:
            param_group['lr'] = min_lr + (max_lr - min_lr) * rho

        self.n_batches_ += 1
