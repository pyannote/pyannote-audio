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
from dlib import probability_that_sequence_is_increasing
from .callback import Callback
from tqdm import tqdm
from scipy.signal import convolve


AUTO_LR_MIN = 1e-6
AUTO_LR_MAX = 1e3
AUTO_LR_BATCHES = 500

MOMENTUM_MAX = 0.95
MOMENTUM_MIN = 0.85

class BaseSchedulerCallback(Callback):
    """Base scheduler with support for AutoLR


    Reference
    ---------
    Leslie N. Smith. "Cyclical Learning Rates for Training Neural Networks"
    IEEE Winter Conference on Applications of Computer Vision (WACV, 2017).

    """

    def on_train_start(self, trainer):
        self.optimizer_ = trainer.optimizer_
        if trainer.base_learning_rate_ == 'auto':
            trainer.base_learning_rate_ = self.auto_lr(trainer)
        self.learning_rate = trainer.base_learning_rate_

    def on_epoch_start(self, trainer):
        trainer.tensorboard_.add_scalar(
            f'train/lr', self.learning_rate,
            global_step=trainer.epoch_)

    def learning_rate():
        doc = "Learning rate."
        def fget(self):
            return self._learning_rate
        def fset(self, lr):
            for g in self.optimizer_.param_groups:
                g['lr'] = lr
                g['momentum'] = MOMENTUM_MAX
            self._learning_rate = lr
        return locals()
    learning_rate = property(**learning_rate())

    def choose_lr(self, lrs, losses):

        min_lr = np.min(lrs)
        max_lr = np.max(lrs)
        n_batches = len(lrs)

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        factor = (max_lr / min_lr) ** (1 / n_batches)

        # `K` batches to increase the learning rate by one order of magnitude
        K = int(np.log(10) / np.log(factor))

        losses = convolve(losses, 3 * np.ones(K // 3) / K,
                          mode='same', method='auto')

        # probability that loss has decreased in the last `K` steps.
        probability = [probability_that_sequence_is_increasing(-losses[i-K:i])
                       if i > K else np.NAN for i in range(len(losses))]
        probability = np.array(probability)

        # find longest decreasing region
        decreasing = 1 * (probability > 0.999)
        starts_decreasing = np.where(np.diff(decreasing) == 1)[0]
        stops_decreasing = np.where(np.diff(decreasing) == -1)[0]
        i = np.argmax(
            [stop - start for start, stop in zip(starts_decreasing,
                                                 stops_decreasing)])
        stop = stops_decreasing[i]

        # upper bound
        # heuristic: loss ceased to decrease between stop-K and stop
        # so we'd rather bound the learning rate slighgly before stop - K
        return lrs[int(stop - 1.1 * K)]

    def auto_lr(self, trainer, beta=0.98):

        trainer.save_epoch()

        # initialize optimizer with a low learning rate
        for param_group in trainer.optimizer_.param_groups:
            param_group['lr'] = AUTO_LR_MIN

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        factor = (AUTO_LR_MAX / AUTO_LR_MIN) ** (1 / 500)

        # progress bar
        pbar = tqdm(
            desc='AutoLR',
            total=AUTO_LR_BATCHES,
            leave=False,
            ncols=80,
            unit='batch',
            position=1)

        loss_moving_avg = 0.
        losses, losses_smoothened, lrs = [], [], []

        # loop on n_batches batches
        for i in range(AUTO_LR_BATCHES):

            batch = next(trainer.batches_)
            loss = trainer.batch_loss(batch)['loss']
            loss.backward()

            # TODO. use closure
            # https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            trainer.optimizer_.step()
            trainer.optimizer_.zero_grad()

            lrs.append(trainer.optimizer_.param_groups[0]['lr'])

            loss = loss.detach().cpu().item()
            losses.append(loss)

            loss_moving_avg = beta * loss_moving_avg + (1 - beta) * loss
            losses_smoothened.append(loss_moving_avg / (1 - beta ** (i + 1)))

            # update progress bar
            pbar.update(1)
            pbar.set_postfix(
                ordered_dict={'loss': losses_smoothened[-1], 'lr': lrs[-1]})

            # increase learning rate
            for param_group in trainer.optimizer_.param_groups:
                param_group['lr'] *= factor

            # stop AutoLR early when loss starts to explode
            if i > 1 and losses_smoothened[-1] > 100 * np.nanmin(losses_smoothened):
                break

        # reload model using its initial state
        trainer.load_epoch(trainer.epoch_)

        lr = self.choose_lr(lrs, losses_smoothened)

        try:

            # import matplotlib with headless backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # create AutoLR loss = f(learning_rate) curve
            fig, ax = plt.subplots()
            ax.semilogx(lrs, losses, '.', alpha=0.3, label='Raw loss')
            ax.semilogx(lrs, losses_smoothened, linewidth=2, label='Smoothened loss')
            ax.set_xlabel('Learning rate')
            ax.set_ylabel('Loss')
            ax.legend()

            # indicate selected learning rate by a vertical line
            ax.plot([lr, lr], [np.nanmin(losses_smoothened),
                               np.nanmax(losses_smoothened)],
                               linewidth=3)

            # zoom on meaningful part of the curve
            m = np.nanmin(losses_smoothened)
            M = 1.1 * losses_smoothened[10]
            ax.set_ylim(m, M)

            # indicate selected learning rate in the figure title
            ax.set_title(f'AutoLR = {lr:g}')

            # send matplotlib figure to Tensorboard
            trainer.tensorboard_.add_figure(
                'train/auto_lr', fig,
                global_step=trainer.epoch_,
                close=True)

        except ImportError as e:
            msg = (
                'Something went wrong when trying to send AutoLR figure '
                'to Tensorboard. Did you install matplotlib?'
            )
            print(msg)

        except Exception as e:
            msg = (
                'Something went wrong when trying to send AutoLR figure '
                'to Tensorboard. It is OK but you might want to have a '
                'look at why this happened...'
            )
            print(msg)

        return lr


class ConstantScheduler(BaseSchedulerCallback):
    """Constant learning rate"""
    pass


class DavisKingScheduler(BaseSchedulerCallback):
    """Automatic Learning Rate Scheduling That Really Works

    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

    Parameters
    ----------
    factor : float, optional
        Factor by which the learning rate will be reduced.
        new_lr = old_lr * factor. Defaults to 0.5
    patience : int, optional
        Number of epochs with no improvement after which learning rate will
        be reduced. Defaults to 10.
    """

    def __init__(self, factor=0.5, patience=10):
        super().__init__()
        self.factor = factor
        self.patience = patience

    def on_train_start(self, trainer):
        super().on_train_start(trainer)
        maxlen = 10 * self.patience * trainer.batches_per_epoch_
        self.losses_ = deque([], maxlen=maxlen)

    def on_batch_end(self, trainer, batch_loss):
        super().on_batch_end(trainer, batch_loss)

        # store current batch loss
        loss = batch_loss['loss'].detach().cpu().item()
        self.losses_.append(loss)

        # compute statistics on batch loss trend
        count = count_steps_without_decrease(self.losses_)
        count_robust = count_steps_without_decrease_robust(self.losses_)

        # if batch loss hasn't been decreasing for a while
        patience = self.patience * trainer.batches_per_epoch_
        if count > patience and count_robust > patience:
            self.learning_rate = self.factor * self.learning_rate
            self.losses_.clear()


class CyclicScheduler(BaseSchedulerCallback):
    """Cyclic learning rate (and momentum)

    Parameters
    ----------
    epochs_per_cycle : int, optional
        Number of epochs per cycle. Defaults to 20.
    decay : {float, 'auto'}, optional
        Update base learning rate at the end of each cycle:
            - when `float`, multiply base learning rate by this amount;
            - when 'auto', apply AutoLR;
            - defaults to doing nothing.


    Reference
    ---------
    Leslie N. Smith. "Cyclical Learning Rates for Training Neural Networks"
    IEEE Winter Conference on Applications of Computer Vision (WACV, 2017).
    """

    def __init__(self, epochs_per_cycle=20, decay=None):
        super().__init__()
        self.epochs_per_cycle = epochs_per_cycle
        self.decay = decay

    def momentum():
        doc = "Momentum."
        def fget(self):
            return self._momentum
        def fset(self, momentum):
            for g in self.optimizer_.param_groups:
                g['momentum'] = momentum
            self._momentum = momentum
        return locals()
    momentum = property(**momentum())

    def on_train_start(self, trainer):
        """Initialize batch/epoch counters"""

        super().on_train_start(trainer)
        self.batches_per_cycle_ = \
            self.epochs_per_cycle * trainer.batches_per_epoch_
        self.n_batches_ = 0
        self.n_epochs_ = 0

        self.learning_rate = trainer.base_learning_rate_ * 0.1

    def on_epoch_end(self, trainer):
        """Update base learning rate at the end of cycle"""

        super().on_epoch_end(trainer)

        # reached end of cycle?
        self.n_epochs_ += 1
        if self.n_epochs_ % self.epochs_per_cycle == 0:

            # apply AutoLR
            if self.decay == 'auto':
                trainer.base_learning_rate_ = self.auto_lr(trainer)

            # decay base learning rate
            elif self.decay is not None:
                trainer.base_learning_rate_ *= self.decay

            # reset epoch/batch counters
            self.n_epochs_ = 0
            self.n_batches_ = 0

    def on_batch_start(self, trainer, batch):
        """Update learning rate & momentum according to position in cycle"""

        super().on_batch_start(trainer, batch)

        # position within current cycle (reversed V)
        rho = 1. - abs(2 * (self.n_batches_ / self.batches_per_cycle_ - 0.5))

        self.learning_rate = trainer.base_learning_rate_ * (0.1 + 0.9 * rho)
        self.momentum = MOMENTUM_MAX - (MOMENTUM_MAX - MOMENTUM_MIN) * rho

        self.n_batches_ += 1

        return batch
