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
from torch.optim import Adam
from torch.optim import RMSprop
from abc import ABCMeta, abstractmethod
from pyannote.audio.checkpoint import Checkpoint
from tensorboardX import SummaryWriter
from dlib import count_steps_without_decrease
from dlib import count_steps_without_decrease_robust
from dlib import probability_that_sequence_is_increasing


class DavisKingScheduler(object):
    """Automatic Learning Rate Scheduling That Really Works

    http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    batches_per_epoch : int
        Number of batches per epoch.
    factor : float, optional
        Factor by which the learning rate will be reduced.
        new_lr = old_lr * factor. Defaults to 0.9
    patience : int, optional
        Number of epochs with no improvement after which learning rate will
        be reduced. Defaults to 20.

    Example
    -------
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> batches_per_epoch = 1000
    >>> scheduler = DavisKingScheduler(optimizer, batches_per_epoch)
    >>> for mini_batch in batches:
    ...     mini_loss = train(mini_batch, optimizer)
    ...     scheduler.step(mini_loss)
    """

    def __init__(self, optimizer, batches_per_epoch, factor=0.9,
                 patience=20):

        super(DavisKingScheduler, self).__init__()
        self.batches_per_epoch = batches_per_epoch
        self.optimizer = optimizer
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
                old_lr = param_group['lr']
                param_group['lr'] = old_lr * self.factor

            # reset batch loss trend
            self.batch_losses_.clear()

        return {
            'epochs_without_decrease': count / self.batches_per_epoch,
            'epochs_without_decrease_robust': \
                count_robust / self.batches_per_epoch}




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

    def fit(self, model, feature_extraction, protocol,
            log_dir=None, subset='train', epochs=1000,
            restart=0, device=None):
        """Train model

        Parameters
        ----------
        model : torch.nn.Module
            Sequence labeling model.
        feature_extraction : pyannote.audio.features.Precomputed
            Precomputed features.
        protocol : pyannote.database.Protocol
            Evaluation protocol.
        subset : {'train', 'development', 'test'}, optional
            Subset to use for training. Defaults to "train".
        log_dir : str, optional
            Directory where models and other log files are stored.
            Defaults to not store anything.
        epochs : int, optional
            Train model for that many epochs. Defaults to 1000.
        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        device : torch.device, optional
            Defaults to torch.device('cpu')

        Returns
        -------
        model : torch.nn.Module
            Trained model.
        """

        iterations = self.fit_iter(model, feature_extraction,
                                   protocol, log_dir=log_dir,
                                   subset=subset, epochs=epochs,
                                   restart=restart, device=device)

        for iteration in iterations:
            pass

        return iteration['model']


    def fit_iter(self, model, feature_extraction,
                 protocol, subset='train',
                 epochs=1000, restart=0,
                 log_dir=None, device=None, quiet=False):

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

        optimizer = SGD(model.parameters(),
                        lr=self.learning_rate,
                        momentum=0.9,
                        dampening=0,
                        weight_decay=0,
                        nesterov=True)

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

        # Davis King's scheduler "that just works"
        scheduler = DavisKingScheduler(optimizer, batches_per_epoch,
                                       factor=0.5, patience=20)

        self.on_train_start(model, batches_per_epoch=batches_per_epoch)

        epoch = restart if restart > 0 else -1
        iteration = -1

        # Buffer containing batch losses for last few epochs
        # It will be used at the end of each epoch to detect
        # whenever the loss is exploding. When that happens,
        # backtracking is triggered to revert the model back
        # to a previous state where it was still going fine.
        backtrack_patience = 3
        batch_losses = deque([], backtrack_patience * batches_per_epoch)

        epoch_pbar = tqdm(desc='Iteration', total=epochs - restart,
                          position=0, unit='epoch',
                          postfix={'loss': '...', 'epoch': '...'})
        batch_pbar = tqdm(desc='Batch', total=batches_per_epoch,
                          position=1, unit='batch',
                          postfix={'loss': '...'})

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
                # 'scheduler_state' is a dictionary that
                # is logged to tensorboard at each epoch.
                scheduler_state = scheduler.batch_step(loss.item())

                batch_pbar.set_postfix(ordered_dict={'loss': loss.item()})
                batch_pbar.update(1)

            # tensorboard: average loss
            loss_avg /= batches_per_epoch
            writer.add_scalar('train/loss', loss_avg, global_step=iteration)

            epoch_pbar.set_postfix(
                ordered_dict={'loss': loss_avg, 'epoch': epoch})
            epoch_pbar.update(1)

            batch_pbar.close()
            batch_pbar = tqdm(desc='Batch', total=batches_per_epoch,
                              position=1, unit='batch',
                              postfix={'loss': '...'})

            # tensorboard: learning rate
            writer.add_scalar('train/learning_rate',
                              optimizer.param_groups[0]['lr'],
                              global_step=iteration)

            # tensorboard: scheduler
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
