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

import io
import yaml
import torch
import tempfile
from torch.optim import SGD
from pyannote.audio.train.schedulers import ConstantScheduler
from pyannote.audio.train.checkpoint import Checkpoint
from tensorboardX import SummaryWriter
from .logging import Logging
from .auto_lr import AutoLR
from .callback import Callbacks

ARBITRARY_LR = 0.1


class Trainer:
    """Trainer"""

    SPECS_YML = '{log_dir}/weights/specs.yml'
    WEIGHTS_DIR = '{log_dir}/weights'
    WEIGHTS_PT = '{log_dir}/weights/{epoch:04d}.pt'
    OPTIMIZER_PT = '{log_dir}/weights/{epoch:04d}.optimizer.pt'

    def load(self, epoch):

        # TODO. check that model specs are coherent

        # load model
        model_state = torch.load(
            self.WEIGHTS_PT.format(log_dir=self.log_dir_, epoch=epoch),
            map_location=lambda storage, loc: storage)
        self.model_.load_state_dict(model_state)

        # load optimizer
        optimizer_state = torch.load(
            self.OPTIMIZER_PT.format(log_dir=self.log_dir_, epoch=epoch),
            map_location=lambda storage, loc: storage)
        self.optimizer_.load_state_dict(optimizer_state)

        self.epoch_ = epoch

    def save(self):

        torch.save(self.model_.state_dict(),
                   self.WEIGHTS_PT.format(log_dir=self.log_dir_,
                                          epoch=self.epoch_))
        torch.save(self.optimizer_.state_dict(),
                   self.OPTIMIZER_PT.format(log_dir=self.log_dir_,
                                            epoch=self.epoch_))

    def on_train_start(self):
        pass

    def fit(self, model, batch_generator, restart=0, epochs=1000,
            get_optimizer=None, get_scheduler=None, learning_rate='auto',
            log_dir=None, device=None):
        """Train model

        Parameters
        ----------
        model : torch.nn.Module
            Sequence labeling/embedding model.
        batch_generator : `callable`

        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        epochs : int, optional
            Train model for that many epochs. Defaults to 1000.
        get_optimizer : callable, optional
            Function that takes `model.parameters()` and `lr=...` as input and
            returns an optimizer. Defaults to `torch.optim.SGD`.
        get_scheduler : callable, optional
            Function that takes `optimizer`, `batches_per_epoch`, `min_lr=...`,
            and `max_lr=...` as input and returns a learning rate scheduler.
            Defaults to `pyannote.audio.train.schedulers.ConstantScheduler`.
        learning_rate : {float, 'auto'}, optional
            Defaults to 'auto'.
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
            model, batch_generator,
            restart=restart, epochs=epochs,
            get_optimizer=get_optimizer, get_scheduler=get_scheduler,
            learning_rate=learning_rate, log_dir=log_dir, device=device)

        for _ in iterations:
            pass

        return self.model_

    def fit_iter(self, get_model, batch_generator,
                 restart=0, epochs=1000,
                 get_optimizer=None, get_scheduler=None, learning_rate='auto',
                 log_dir=None, device=None):
        """Train model

        Parameters
        ----------
        get_model : callable
            Callable that takes batch generator specification as input and
            returns a nn.Module instance
        batch_generator : callable

        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        epochs : int, optional
            Train model for that many epochs. Defaults to 1000.
        get_optimizer : callable, optional
            Function that takes `model.parameters()` and `lr=...` as input and
            returns an optimizer. Defaults to `torch.optim.SGD`.
        get_scheduler : callable, optional
            Function that takes `optimizer`, `batches_per_epoch`, `min_lr=...`,
            and `max_lr=...` as input and returns a learning rate scheduler.
            Defaults to `pyannote.audio.train.schedulers.ConstantScheduler`.
        learning_rate : {float, 'auto'}, optional
            Defaults to 'auto'.
        log_dir : str, optional
            Directory where models and other log files are stored.
            Defaults to not store anything.
        device : torch.device, optional
            Defaults to torch.device('cpu')

        Yields
        ------
        result : dict
            ['epoch'] (int): current epoch
            ['iteration'] (int): current iteration
            ['model']  (torch.nn.Module): model at current iteration
        """

        # LOGGING
        if log_dir is None:
            self.log_dir_ = tempfile.mkdtemp()
        else:
            self.log_dir_ = log_dir
        self.tensorboard_ = SummaryWriter(log_dir=self.log_dir_)

        # BATCH GENERATOR
        self.batch_generator_ = batch_generator
        self.batches_ = self.batch_generator_()
        self.batches_per_epoch_ = self.batch_generator_.batches_per_epoch

        # DEVICE
        self.device_ = torch.device('cpu') if device is None else device

        # MODEL
        specifications = self.batch_generator_.specifications
        self.model_ = get_model(specifications)
        specs_yml = self.SPECS_YML.format(log_dir=self.log_dir_)
        with io.open(specs_yml, 'w') as fp:
            yaml.dump(specifications, fp,
                      default_flow_style=False)
        self.model_ = self.model_.to(self.device_)

        # OPTIMIZER
        if get_optimizer is None:
            get_optimizer = SGD
        lr = ARBITRARY_LR if learning_rate == 'auto' else learning_rate
        self.optimizer_ = get_optimizer(self.model_.parameters(), lr=lr)

        # SCHEDULER
        if get_scheduler is None:
            get_scheduler = ConstantScheduler
        self.scheduler_ = get_scheduler(min_lr=None, max_lr=lr)

        callbacks = [
            Checkpoint(),  # checkpoint has to go first
            Logging(epochs),
            self.scheduler_,
        ]

        # AUTOMATIC LEARNING RATE
        if learning_rate == 'auto':
            callbacks.append(AutoLR())

        callbacks = Callbacks(callbacks)

        if restart:
            callbacks.load_epoch(self, restart)
        else:
            self.epoch_ = 0

        callbacks.on_train_start(self)

        while self.epoch_ < epochs:

            callbacks.on_epoch_start(self)

            for i in range(self.batches_per_epoch_):
                batch = next(self.batches_)

                callbacks.on_batch_start(self, batch)

                loss = self.batch_loss(batch)
                loss.backward()
                self.optimizer_.step()
                self.optimizer_.zero_grad()

                loss = loss.detach().cpu().item()
                callbacks.on_batch_end(self, loss)

            callbacks.on_epoch_end(self)

            yield self.model_

        callbacks.on_train_end(self)
