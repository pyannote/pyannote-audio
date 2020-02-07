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
import scipy.stats
from collections import deque
from .callback import Callback
from tqdm import tqdm
from scipy.signal import convolve

from typing import TYPE_CHECKING

AUTOLR_MIN = 0.000001
AUTOLR_MAX = 10
AUTOLR_EPOCHS = 8
AUTOLR_BETA = 0.98

MOMENTUM_MAX = 0.95
MOMENTUM_MIN = 0.85


def decreasing_probability(values: np.ndarray) -> float:
    """Compute probability that a sequence is decreasing

    Parameters
    ----------
    values : np.ndarray
        Sequence of values

    Returns
    -------
    probability : float
        Probability that sequence of values is decreasing

    Reference
    ---------
    Davis King. "Automatic Learning Rate Scheduling That Really Works".
    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
    """
    n_steps = len(values)
    steps = np.arange(n_steps)

    A = np.vstack([steps, np.ones(n_steps)]).T
    loc, shift = np.linalg.lstsq(A, values, rcond=None)[0]

    values_ = loc * steps + shift
    sigma2 = np.sum((values - values_) ** 2) / (n_steps - 2)

    scale = np.sqrt(12 * sigma2 / (n_steps ** 3 - n_steps))
    return scipy.stats.norm.cdf(0., loc=loc, scale=scale)


def steps_without_decrease(values: np.ndarray,
                           robust: bool = False) -> int:
    """Count number of steps without decrease

    Parameters
    ----------
    values : np.ndarray
        Sequence of values
    robust : bool
        Remove 10% highest values before counting steps.

    Returns
    -------
    n_steps : int
        Number of steps

    Reference
    ---------
    Davis King. "Automatic Learning Rate Scheduling That Really Works".
    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html
    """

    if robust:
        values = values[values < np.percentile(values, 90)]

    steps_without_decrease = 0
    n_steps = len(values)
    for i in reversed(range(n_steps - 2)):
        p = decreasing_probability(values[i:])
        if p < 0.51:
            steps_without_decrease = n_steps - i
    return steps_without_decrease


class BaseSchedulerCallback(Callback):
    """Base scheduler with support for AutoLR

    Reference
    ---------
    Leslie N. Smith. "Cyclical Learning Rates for Training Neural Networks"
    IEEE Winter Conference on Applications of Computer Vision (WACV, 2017).
    """

    def on_train_start(self, trainer: 'Trainer') -> None:
        self.optimizer_ = trainer.optimizer
        if trainer.base_learning_rate_ == 'auto':
            trainer.base_learning_rate_ = self.auto_lr(trainer)
        self.learning_rate = trainer.base_learning_rate_

    def on_epoch_start(self, trainer: 'Trainer') -> None:
        """Log learning rate to tensorboard"""

        trainer.tensorboard_.add_scalar(
            f'train/lr', self.learning_rate,
            global_step=trainer.epoch_)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        for g in self.optimizer_.param_groups:
            g['lr'] = learning_rate
            g['momentum'] = MOMENTUM_MAX
        self._learning_rate = learning_rate

    @staticmethod
    def _choose_lr(lrs: np.ndarray, losses: np.ndarray) -> float:
        """Choose learning rate upper bound

        Parameters
        ----------
        lrs : np.ndarray
            Sequence of learning rates
        losses : np.ndarray
            Corresponding sequence of loss values

        Returns
        -------
        max_lr : float
            Return learning rate upper bound
        """

        min_lr, max_lr = np.min(lrs), np.max(lrs)
        n_batches = len(lrs)

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        factor = (max_lr / min_lr) ** (1 / n_batches)

        # `K` batches to increase the learning rate by one order of magnitude
        K = int(np.log(10) / np.log(factor))

        losses = convolve(losses, 3 * np.ones(K // 3) / K,
                          mode='same', method='auto')

        # probability that loss has decreased in the last `K` steps.
        probability = [1. - decreasing_probability(-losses[i-K:i])
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

    def auto_lr(self, trainer: 'Trainer') -> float:
        """Find optimal learning rate automatically

        Parameters
        ----------
        trainer : 'Trainer'

        Returns
        -------
        learning_rate : float
            Optimal learning rate

        Reference
        ---------
        Leslie N. Smith. "Cyclical Learning Rates for Training Neural Networks"
        IEEE Winter Conference on Applications of Computer Vision (WACV, 2017).

            There is a simple way to estimate reasonable minimum and maximum
            boundary values with one training run of the network for a few
            epochs. It is a "LR range test"; run your model for several epochs
            while letting the learning rate increase linearly between low and
            high LR values. This test is enormously valuable whenever you are
            facing a new architecture or dataset. Next, plot the accuracy
            versus learning rate. Note the learning rate value when the
            accuracy starts to increase and when the accuracy slows, becomes
            ragged, or starts to fall. These two learning rates are good
            choices for bounds; that is, set base_lr to the first value and
            set max_lr to the latter value. Alternatively, one can use the rule
            of thumb that the optimum learning rate is usually within a factor
            of two of the largest one that converges and set base_lr to 1/3 or
            1/4 of max_lr. [...] Whenever one is starting with a new
            architecture or dataset, a single LR range test provides both a
            good LR value and a good range. Then one should compare runs with a
            fixed LR versus CLR with this range. Whichever wins can be used
            with confidence for the rest of one’s experiments.
        """
        # save states to disk
        trainer.save_state()

        # initialize optimizer with a low learning rate
        self.learning_rate = AUTOLR_MIN

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        n_batches: int = AUTOLR_EPOCHS * trainer.batches_per_epoch
        factor = (AUTOLR_MAX / AUTOLR_MIN) ** (1 / n_batches)

        # progress bar
        pbar = tqdm(
            desc='AutoLR',
            total=n_batches,
            leave=False,
            ncols=80,
            unit='batch',
            position=1)

        loss_moving_avg = 0.
        losses, losses_smoothened, lrs = [], [], []

        # loop on n_batches batches
        for i in range(n_batches):

            batch = trainer.get_new_batch()
            loss = trainer.batch_loss(batch)
            loss['loss'].backward()
            trainer.optimizer_.step()
            trainer.optimizer_.zero_grad()

            l = loss['loss'].item()
            if np.isnan(l):
                break

            losses.append(l)
            lrs.append(self.learning_rate)

            loss_moving_avg = AUTOLR_BETA * loss_moving_avg + \
                (1 - AUTOLR_BETA) * losses[-1]
            losses_smoothened.append(
                loss_moving_avg / (1 - AUTOLR_BETA ** (i + 1)))

            # update progress bar
            pbar.update(1)
            pbar.set_postfix(
                ordered_dict={'loss': losses_smoothened[-1],
                              'lr': self.learning_rate})

            # increase learning rate
            self.learning_rate = factor * self.learning_rate

            # stop AutoLR early when loss starts to explode
            if i > 1 and losses_smoothened[-1] > 100 * np.nanmin(losses_smoothened):
                break

        # reload model using its initial state
        trainer.load_state()

        # choose learning rate based on loss = f(learning_rate) curve
        auto_lr = self._choose_lr(lrs, losses_smoothened)

        # log curve and auto_lr to tensorboard as an image
        try:
            # import matplotlib with headless backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # create AutoLR loss = f(learning_rate) curve
            fig, ax = plt.subplots()
            ax.semilogx(lrs, losses, '.', alpha=0.3, label='Raw loss')
            ax.semilogx(lrs, losses_smoothened, linewidth=2,
                        label='Smoothened loss')
            ax.set_xlabel('Learning rate')
            ax.set_ylabel('Loss')
            ax.legend()

            # indicate selected learning rate by a vertical line
            ax.plot(
                [auto_lr, auto_lr],
                [np.nanmin(losses_smoothened), np.nanmax(losses_smoothened)],
                linewidth=3)

            # zoom on meaningful part of the curve
            m = np.nanmin(losses_smoothened)
            M = 1.1 * losses_smoothened[10]
            ax.set_ylim(m, M)

            # indicate selected learning rate in the figure title
            ax.set_title(f'AutoLR = {auto_lr:g}')

            # send matplotlib figure to Tensorboard
            trainer.tensorboard_.add_figure(
                'train/auto_lr', fig,
                global_step=trainer.epoch_,
                close=True)

        except ImportError as e:
            msg = (
                f'Something went wrong when trying to send AutoLR figure '
                f'to Tensorboard. Did you install matplotlib?\n\n{e}\n\n'
            )
            print(msg)
            print(e)

        except Exception as e:
            msg = (
                f'Something went wrong when trying to send AutoLR figure '
                f'to Tensorboard. It is OK but you might want to have a '
                f'look at why this happened.\n\n{e}\n\n'
            )
            print(msg)

        return auto_lr


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

    def __init__(self, factor: float = 0.5, patience: int = 10):
        super().__init__()
        self.factor = factor
        self.patience = patience

    def on_train_start(self, trainer: 'Trainer') -> None:
        super().on_train_start(trainer)
        maxlen = 2 * self.patience * trainer.batches_per_epoch
        self.losses_ = deque([], maxlen=maxlen)

    def on_epoch_end(self, trainer: 'Trainer') -> None:

        # compute statistics on batch loss trend
        count = steps_without_decrease(
            np.array(self.losses_))
        count_robust = steps_without_decrease(
            np.array(self.losses_), robust=True)

        # if batch loss hasn't been decreasing for a while
        patience = self.patience * trainer.batches_per_epoch
        if count > patience and count_robust > patience:
            self.learning_rate = self.factor * self.learning_rate
            self.losses_.clear()

    def on_batch_end(self, trainer, batch_loss):
        super().on_batch_end(trainer, batch_loss)

        # store current batch loss
        self.losses_.append(batch_loss['loss'].item())


class CyclicScheduler(BaseSchedulerCallback):
    """Cyclic learning rate (and momentum)

    Parameters
    ----------
    epochs_per_cycle : int, optional
        Number of epochs per cycle. Defaults to 10.
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

    def __init__(self, epochs_per_cycle: int = 10,
                       decay=None):
        super().__init__()
        self.epochs_per_cycle = epochs_per_cycle
        self.decay = decay

    @property
    def momentum(self) -> float:
        return self._momentum

    @momentum.setter
    def momentum(self, momentum: float) -> None:
        for g in self.optimizer_.param_groups:
            g['momentum'] = momentum
        self._momentum = momentum

    def on_train_start(self, trainer: 'Trainer') -> None:
        """Initialize batch/epoch counters"""

        super().on_train_start(trainer)
        self.batches_per_cycle_ = \
            self.epochs_per_cycle * trainer.batches_per_epoch
        self.n_batches_ = 0
        self.n_epochs_ = 0

        self.learning_rate = trainer.base_learning_rate_ * 0.1

    def on_epoch_end(self, trainer: 'Trainer') -> None:
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

    def on_batch_start(self, trainer: 'Trainer', batch: dict) -> dict:
        """Update learning rate & momentum according to position in cycle"""

        super().on_batch_start(trainer, batch)

        # position within current cycle (reversed V)
        rho = 1. - abs(2 * (self.n_batches_ / self.batches_per_cycle_ - 0.5))

        self.learning_rate = trainer.base_learning_rate_ * (0.1 + 0.9 * rho)
        self.momentum = MOMENTUM_MAX - (MOMENTUM_MAX - MOMENTUM_MIN) * rho

        self.n_batches_ += 1

        return batch
