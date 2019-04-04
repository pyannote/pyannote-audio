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


from .callback import Callback
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve
from dlib import probability_that_sequence_is_increasing


class AutoLR(Callback):
    """Automagically find a "good" learning rate upper bound

    Parameters
    ----------
    min_lr, max_lr : float, optional
        Learning rate will be increased exponentially from `min_lr` to
        `max_lr`.
    n_batches : int, optional
        Number of batches needed to increase from `min_lr` to `max_lr`.

    Reference
    ---------
    https://gist.github.com/hbredin/1d617586017837acb35b090f50f7a22b
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    http://brandonlmorris.com/2018/06/24/mastering-the-learning-rate
    """

    def __init__(self, min_lr=1e-6, max_lr=1e3, n_batches=500):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_batches = n_batches

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
        upper = lrs[int(stop - 1.1 * K)]

        # lower bound. make sure there is at least one order of magnitude
        # between lower and upper bounds
        lower = 0.1 * upper

        return lower, upper

    def on_train_start(self, trainer):

        trainer.save()

        # initialize optimizer with a low learning rate
        for param_group in trainer.optimizer_.param_groups:
            param_group['lr'] = self.min_lr

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        factor = (self.max_lr / self.min_lr) ** (1 / self.n_batches)

        self.batches_pbar_ = tqdm(
            desc=f'Epoch #{trainer.epoch_}',
            total=trainer.batches_per_epoch_, leave=False, ncols=80,
            unit='batch', position=1)


        # progress bar
        pbar = tqdm(
            desc='AutoLR',
            total=self.n_batches,
            leave=False,
            ncols=80,
            unit='batch',
            position=1)

        losses, lrs = [], []

        # loop on n_batches batches
        for i in range(self.n_batches):

            batch = next(trainer.batches_)
            loss = trainer.batch_loss(batch)
            loss.backward()

            # TODO. use closure
            # https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            trainer.optimizer_.step()
            trainer.optimizer_.zero_grad()

            lrs.append(trainer.optimizer_.param_groups[0]['lr'])
            losses.append(loss.detach().cpu().item())

            # update progress bar
            pbar.update(1)
            pbar.set_postfix(
                ordered_dict={'loss': losses[-1], 'lr': lrs[-1]})

            trainer.tensorboard_.add_scalar(
                f'auto_lr/loss', losses[-1], global_step=i)
            trainer.tensorboard_.add_scalar(
                f'auto_lr/lr', lrs[-1], global_step=i)

            # increase learning rate
            for param_group in trainer.optimizer_.param_groups:
                param_group['lr'] *= factor

            # stop AutoLR early when loss starts to explode
            if i > 1 and losses[-1] > 100 * np.nanmin(losses):
                break

        # reload model using its initial state
        trainer.load(trainer.epoch_)

        min_lr, max_lr = self.choose_lr(lrs, losses)

        trainer.scheduler_.min_lr = min_lr
        trainer.scheduler_.max_lr = max_lr
