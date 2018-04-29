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
    patience_down : int, optional
        Number of epochs with no improvement after which learning rate will
        be reduced. Defaults to 50.
    patience_up : int, optional
        Defaults to 5.
    active : bool, optional
        Set to False to not update learning rate.

    Usage
    -----
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> batches_per_epoch = 1000
    >>> scheduler = DavisKingScheduler(optimizer, batches_per_epoch)
    >>> for mini_batch in batches:
    ...     mini_loss = train(mini_batch, optimizer)
    ...     scheduler.step(mini_loss)
    """

    def __init__(self, optimizer, batches_per_epoch, factor=0.9,
                 patience_down=50, patience_up=5, active=True):

        super(DavisKingScheduler, self).__init__()

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.optimizer = optimizer
        self.patience_down = patience_down
        self.patience_up = patience_up
        self.batches_per_epoch = batches_per_epoch
        self.active = active

        self.lr_ = [float(grp['lr']) for grp in self.optimizer.param_groups]
        patience = max(self.patience_down, self.patience_up)
        self.losses_ = deque([], maxlen=patience * self.batches_per_epoch + 1)

    @property
    def lr(self):
        return tuple(self.lr_)

    def step(self, loss):

        self.losses_.append(loss)

        count = count_steps_without_decrease(self.losses_)
        count_robust = count_steps_without_decrease_robust(self.losses_)

        patience = self.patience_up * self.batches_per_epoch
        if len(self.losses_) > patience:
            # only consider batches from last patience_up epoch
            # https://github.com/davisking/dlib/issues/1257
            losses = list(self.losses_)[-patience:]
            increasing = probability_that_sequence_is_increasing(losses)
        else:
            increasing = -0.1

        patience = self.patience_down * self.batches_per_epoch
        if (self.active and count > patience and count_robust > patience):
            self._reduce_lr()
            self.losses_.clear()

        return {
            'epochs_without_decrease': count / self.batches_per_epoch,
            'epochs_without_decrease_robust': \
                count_robust / self.batches_per_epoch,
            'increasing_probability': increasing}

    def _reduce_lr(self):
        self.lr_ = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            self.lr_.append(new_lr)


class Trainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_batch_generator(self, precomputed):
        """This method should be overriden by subclass

        Parameters
        ----------
        precomputed : `pyannote.audio.features.Precomputed`

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        pass

    @abstractmethod
    def on_train_start(self):
        pass

    @abstractmethod
    def process_batch(self, batch):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch):
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

        log = log_dir is not None

        if log:
            self.checkpoint_ = Checkpoint(
                log_dir, restart=restart > 0)
            self.writer_ = SummaryWriter(log_dir=log_dir)

        self.device_ = torch.device('cpu') if device is None else device

        self.batch_generator_ = self.get_batch_generator(feature_extraction)
        batches = self.batch_generator_(protocol, subset=subset)
        batch = next(batches)
        self.batches_per_epoch_ = self.batch_generator_.batches_per_epoch

        # initialize model, optimizer, and scheduler
        self.model_ = model
        self.backtrack(restart)

        self.on_train_start()

        epoch = restart if restart > 0 else -1
        backtrack, iteration = False, 0

        while True:
            # keep track of actual number of iterations
            iteration += 1

            # keep track of current epoch
            # due to backtracking, this may lag a bit behind `iteration`
            epoch += 1
            if epoch > epochs:
                break

            # log backtracking
            if log:
                self.writer_.add_scalar('train/scheduler/backtrack', epoch,
                                  global_step=iteration)

            # detailed logging to Tensorboard
            # for first 10 epochs then every other 5 epochs
            self.detailed_log_ = log and ((iteration < 10) or (iteration % 5 == 0))

            loss_avg = 0.

            if quiet:
                counter = range(self.batches_per_epoch_)
            else:
                desc = 'Epoch #{0}'.format(epoch)
                counter = tqdm(range(self.batches_per_epoch_), desc=desc)

            for i in counter:
                # zero gradients
                self.model_.zero_grad()

                # process next batch
                batch = next(batches)

                loss = self.process_batch(batch)

                # back-propagation
                loss.backward()

                # gradient descent
                self.optimizer_.step()

                # keep track of loss
                loss_ = loss.item()
                loss_avg += loss_

                # send loss of current batch to scheduler
                # and receive information about loss trend
                scheduler_state = self.scheduler_.step(loss_)

            if log:

                # log loss to tensorboard
                self.writer_.add_scalar('train/loss',
                                        loss_avg / self.batches_per_epoch_,
                                        global_step=iteration)

                # log loss trend statistics to tensorboard
                for name, value in scheduler_state.items():
                    self.writer_.add_scalar(
                        f'train/scheduler/{name}', value,
                        global_step=iteration)

                # log current learning rate to tensorboard
                self.writer_.add_scalar('train/scheduler/lr',
                                        self.scheduler_.lr[0],
                                        global_step=iteration)

                # save model to disk
                self.checkpoint_.on_epoch_end(epoch, self.model_,
                                              self.optimizer_)

            self.on_epoch_end(iteration)

            yield {'epoch': epoch, 'iteration': iteration, 'model': model}

            # backtrack in case loss has increased
            if getattr(self, 'enable_backtrack', False) and \
               scheduler_state['increasing_probability'] > 0.99:
                epoch = max(0, epoch - self.scheduler_.patience_up - 2)
                self.backtrack(epoch)


    def backtrack(self, epoch):
        """Backtrack to `epoch` state

        This assumes that the following attributes have been set already:

        * checkpoint_
        * model_
        * device_
        * batches_per_epoch_

        This will set/update the following hidden attributes:

        * model_
        * optimizer_
        * scheduler_

        """

        if epoch > 0:
            weights_pt = self.checkpoint_.weights_pt(epoch)
            self.model_.load_state_dict(torch.load(weights_pt))

        self.model_ = self.model_.to(self.device_)

        if self.optimizer == 'sgd':
            self.optimizer_ = SGD(self.model_.parameters(),
                                  lr=self.learning_rate,
                                  momentum=0.9, nesterov=True)

        elif self.optimizer == 'adam':
            self.optimizer_ = Adam(self.model_.parameters(),
                                   lr=self.learning_rate)

        elif self.optimizer == 'rmsprop':
            self.optimizer_ = RMSprop(self.model_.parameters(),
                                      lr=self.learning_rate)

        self.model_.internal = False

        if epoch > 0:
            optimizer_pt = self.checkpoint_.optimizer_pt(epoch)
            self.optimizer_.load_state_dict(torch.load(optimizer_pt))
            for state in self.optimizer_.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device_)

        self.scheduler_ = DavisKingScheduler(
            self.optimizer_, self.batches_per_epoch_,
            active=self.optimizer == 'sgd')
