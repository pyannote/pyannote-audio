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


import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from torch.optim import SGD
from scipy.signal import convolve
from abc import ABCMeta, abstractmethod
from pyannote.audio.train.schedulers import DavisKingScheduler
from pyannote.audio.train.checkpoint import Checkpoint
from tensorboardX import SummaryWriter
from dlib import probability_that_sequence_is_increasing

ARBITRARY_LR = 0.1


class Trainer:
    """Trainer"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_batch_generator(self, precomputed):
        """This method should be overriden by subclass

        Parameters
        ----------
        precomputed : :class:`~pyannote.audio.features.Precomputed`

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        pass

    @abstractmethod
    def on_train_start(self, model, **kwargs):
        """This method should be overriden by subclass"""
        pass

    @abstractmethod
    def batch_loss(self, batch, model, device, writer=None, **kwargs):
        """
        Parameters
        ----------
        batch :
        model :
        device :
        writer : SummaryWriter, optional

        Returns
        -------
        loss :
        """
        pass

    @abstractmethod
    def on_epoch_end(self, iteration, writer=None, **kwargs):
        """
        Parameters
        ----------
        iteration : int
        writer : SummaryWriter, optional
        """
        pass

    def to_numpy(self, tensor):
        """Convert torch.Tensor to numpy array"""
        cpu = torch.device('cpu')
        return tensor.detach().to(cpu).numpy()

    def fit(self, model, feature_extraction,
            protocol, subset='train', restart=0, epochs=1000,
            get_optimizer=None, get_scheduler=None, learning_rate='auto',
            log_dir=None, device=None):
        """Train model

        Parameters
        ----------
        model : torch.nn.Module
            Sequence labeling/embedding model.
        feature_extraction : pyannote.audio.features.Precomputed
            Precomputed features.
        protocol : pyannote.database.Protocol
            Evaluation protocol.
        subset : {'train', 'development', 'test'}, optional
            Subset to use for training. Defaults to "train".
        epochs : int, optional
            Train model for that many epochs. Defaults to 1000.
        learning_rate : {float, 'auto'}, optional
            Defaults to 'auto'.
        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        log_dir : str, optional
            Directory where models and other log files are stored.
            Defaults to not store anything.
        device : torch.device, optional
            Defaults to torch.device('cpu')

        Returns
        -------
        model : torch.nn.Module
            Trained model.
        """

        iterations = self.fit_iter(
            model, feature_extraction,
            protocol, subset=subset, restart=restart, epochs=epochs,
            get_optimizer=get_optimizer, get_scheduler=get_scheduler,
            learning_rate=learning_rate, log_dir=log_dir, device=device)

        for iteration in iterations:
            pass

        return iteration['model']

    @staticmethod
    def _choose_lr(lrs, losses, min_lr=1e-6, max_lr=1e3, n_batches=500):
        """Helper function that actually selects the best learning rates

        Parameters
        ----------
        lrs : numpy array
        losses : numpy array
        min_lr, max_lr, n_batches : see `Trainer.auto_lr`

        Returns
        -------
        lower : float
            Learning rate lower bound
        upper : float
            Learning rate upper bound
        """

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

        return {'min_lr': lower,
                'max_lr': upper,
                'lrs': lrs,
                'losses': losses,
                'probability': probability}

    def auto_lr(self, model, optimizer, batches,
                min_lr=1e-6, max_lr=1e3, n_batches=500,
                device=None, writer=None):
        """Automagically find a "good" learning rate upper bound

        Parameters
        ----------
        model : nn.Module

        optimizer : torch.optim.Optimizer
        batches : generator
            Batch generator. `next(batches)` will be called `n_batches` times.
        min_lr, max_lr : float, optional
            Learning rate will be increased exponentially from `min_lr` to
            `max_lr`.
        n_batches : int, optional
            Number of batches needed to increase from `min_lr` to `max_lr`.
        writer : tensorboardX.SummaryWriter, optional
            When provided, log learning rate and loss to tensorboard.
        device : torch.Device, optional
            Device to use. Defaults to `torch.device('cpu')`.

        Returns
        -------
        result : dict
            {'min_lr': <learning rate lower bound>,
             'max_lr': <learning rate upper bound>,
             'lrs': <increasing sequence of learning rates>,
             'losses': <corresponding sequence of loss values>}

        Reference
        ---------
        https://gist.github.com/hbredin/1d617586017837acb35b090f50f7a22b
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        """

        if device is None:
            device = torch.device('cpu')

        self.on_train_start(model, batches_per_epoch=n_batches)

        # initialize optimizer with a low learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = min_lr

        # `factor` by which the learning rate is multiplied after every batch,
        # to get from `min_lr` to `max_lr` in `n_batches` step.
        factor = (max_lr / min_lr) ** (1 / n_batches)

        # progress bar
        pbar = tqdm(desc='Auto LR', total=n_batches,
                    postfix={'loss': '...', 'lr': '...'})

        losses, lrs = [], []

        # loop on n_batches batches
        for i in range(n_batches):

            batch = next(batches)
            model.zero_grad()
            loss = self.batch_loss(batch, model, device)
            loss.backward()
            optimizer.step()

            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

            # update progress bar
            pbar.update(1)
            pbar.set_postfix(ordered_dict={'loss': losses[-1], 'lr': lrs[-1]})

            # update tensorboard
            if writer is not None:
                writer.add_scalar(f'auto_lr/loss', losses[-1], global_step=i)
                writer.add_scalar(f'auto_lr/lr', lrs[-1], global_step=i)

            # increase learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor

        return self._choose_lr(
            np.array(lrs), np.array(losses),
            min_lr=min_lr, max_lr=max_lr, n_batches=n_batches)


    def fit_iter(self, model, feature_extraction,
                 protocol, subset='train', restart=0, epochs=1000,
                 get_optimizer=None, get_scheduler=None, learning_rate='auto',
                 log_dir=None, device=None):

        if log_dir is None and restart > 0:
            msg = ('One must provide `log_dir` when '
                   'using `restart` option.')
            raise ValueError(msg)

        # initialize batch generator
        batch_generator = self.get_batch_generator(feature_extraction)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)
        batches_per_epoch = getattr(batch_generator, 'batches_per_epoch', None)

        # FIXME. log_dir = tmp directory

        checkpoint = Checkpoint(log_dir, restart=restart > 0)
        writer = SummaryWriter(log_dir=log_dir)

        device = torch.device('cpu') if device is None else device
        model = model.to(device)
        optimizer = get_optimizer(model.parameters(), lr=ARBITRARY_LR)

        if restart > 0:

            # load model
            model_state = torch.load(
                checkpoint.weights_pt(restart),
                map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state)

            # load optimizer
            optimizer_state = torch.load(
                checkpoint.optimizer_pt(restart),
                map_location=lambda storage, loc: storage)
            optimizer.load_state_dict(optimizer_state)

        # find optimal learning rate automagically
        if learning_rate == 'auto':

            # save model and optimizer states before "auto_lr"
            if restart == 0:
                checkpoint.on_epoch_end(0, model, optimizer)

            auto_lr = self.auto_lr(model, optimizer, batches,
                                   writer=writer, device=device)
            min_lr = auto_lr['min_lr']
            max_lr = auto_lr['max_lr']

            # dump learning rates and losses to disk for debugging purposes
            with open(f'{log_dir}/auto_lr_log.csv', mode='w') as fp:
                for lr, loss in zip(auto_lr['lrs'], auto_lr['losses']):
                    fp.write(f'{np.log10(lr):g} {loss:g}\n')

            # reload model and optimizer states after "auto_lr"
            model_state = torch.load(
                checkpoint.weights_pt(restart),
                map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state)

            optimizer_state = torch.load(
                checkpoint.optimizer_pt(restart),
                map_location=lambda storage, loc: storage)
            optimizer.load_state_dict(optimizer_state)

        # ... or use the one provided by the user
        else:
            min_lr, max_lr = None, learning_rate

        scheduler = get_scheduler(optimizer, batches_per_epoch,
                                  min_lr=min_lr, max_lr=max_lr)

        self.on_train_start(model, batches_per_epoch=batches_per_epoch)

        epoch = restart if restart > 0 else -1
        iteration = -1

        # Buffer containing batch losses for last few epochs
        # It will be used at the end of each epoch to detect
        # whenever the loss is exploding. When that happens,
        # backtracking is triggered to revert the model back
        # to a previous state where it was still going fine.
        backtrack_patience = 2
        batch_losses = deque([], backtrack_patience * batches_per_epoch)

        while True:

            # keep track of actual number of iterations
            iteration += 1

            # detailed logging to tensorboard
            # for first 10 iterations then every other 5 iteration.
            log = iteration < 10 or iteration % 5 == 0

            # keep track of current epoch
            # due to backtracking, this may lag a bit behind `iteration`
            epoch += 1

            # stop training when that many epochs
            if epoch > epochs:
                break

            # tensorboard: backtracking
            writer.add_scalar('train/backtracking/epoch',
                              epoch, global_step=iteration)

            loss_avg = 0.

            pbar = tqdm(desc=f'Iteration #{iteration}', total=batches_per_epoch,
                        unit='batch', postfix={'loss': '...', 'lr': ...})

            for i in range(batches_per_epoch):

                batch = next(batches)
                model.zero_grad()
                loss = self.batch_loss(batch, model, device,
                                       writer=writer if log else None)
                loss.backward()
                optimizer.step()

                # keep track of loss
                batch_losses.append(loss.item())
                loss_avg += loss.item()

                # send loss of current batch to scheduler
                # 'scheduler_state' is a dictionary that is logged to
                # tensorboard at the end of each epoch.
                scheduler_state = scheduler.batch_step(batch_losses[-1])

                # update progress bar with loss for current batch
                pbar.set_postfix(
                    ordered_dict={'loss': loss_avg / (i+1),
                                  'lr': scheduler_state['lr']})
                pbar.update(1)

            # tensorboard: average loss
            loss_avg /= batches_per_epoch
            writer.add_scalar('train/loss', loss_avg, global_step=iteration)

            # tensorboard: scheduler
            if isinstance(scheduler_state, dict):
                for name, value in scheduler_state.items():
                    writer.add_scalar(
                        f'train/scheduler/{name}', value,
                        global_step=iteration)

            # save model to disk
            checkpoint.on_epoch_end(epoch, model, optimizer)
            # TODO. save scheduler state as well

            self.on_epoch_end(iteration,
                              writer=writer if log else None)

            yield {'epoch': epoch, 'iteration': iteration, 'model': model}

            if not scheduler.allow_backtrack:
                continue

            # tensorboard: backtracking probability
            backtrack_p = probability_that_sequence_is_increasing(batch_losses)
            writer.add_scalar('train/backtracking/probability',
                              backtrack_p, global_step=iteration)

            # backtrack to previous epoch when loss has been increasing
            if backtrack_p > 0.99:

                # backtrack
                epoch = max(0, epoch - backtrack_patience - 2)

                # load model
                model_state = torch.load(
                    checkpoint.weights_pt(epoch),
                    map_location=lambda storage, loc: storage)
                model.load_state_dict(model_state)

                # load optimizer
                optimizer_state = torch.load(
                    checkpoint.optimizer_pt(epoch),
                    map_location=lambda storage, loc: storage)
                optimizer.load_state_dict(optimizer_state)

                # TODO. load scheduler as well
                # scheduler_state = checkpoint.scheduler_pt(epoch)
                # scheduler.load_state_dict(scheduler_state)

                # reset batch loss trend
                batch_losses.clear()
