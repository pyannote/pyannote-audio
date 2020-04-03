#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

from typing import Optional, Iterator, Callable, Union, List

try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal
from pathlib import Path
import io
import os
import sys
import yaml
import torch
import tempfile
import warnings
from itertools import chain
from torch.nn import Module
from torch.optim import Optimizer, SGD
from torch.utils.tensorboard import SummaryWriter
from .logging import Logging
from .callback import Callback
from .callback import Callbacks
from .schedulers import BaseSchedulerCallback
from .schedulers import ConstantScheduler
from .generator import BatchGenerator
from .model import Model
from ..utils.timeout import timeout
from ..utils.background import AdaptiveBackgroundGenerator


ARBITRARY_LR = 0.1


class Trainer:
    """Trainer

    Attributes
    ----------
    model : Model
    specifications : dict
    device : torch.device
    """

    SPECS_YML = "{train_dir}/specs.yml"
    MODEL_PT = "{train_dir}/weights/{epoch:04d}.pt"
    OPTIMIZER_PT = "{train_dir}/weights/{epoch:04d}.optimizer.pt"

    def load_state(self, model_pt: Optional[Path] = None) -> bool:
        """Load model and optimizer states from disk

        Parameters
        ----------
        model_pt : `Path`, optional
            Path to file containing model state.
            Defaults to guessing it from trainer status.

        Returns
        -------
        success : bool
            True if state was loaded successfully, False otherwise.
        """

        if model_pt is None:
            _model_pt = self.MODEL_PT.format(
                train_dir=self.train_dir_, epoch=self.epoch_
            )
            optimizer_pt = self.OPTIMIZER_PT.format(
                train_dir=self.train_dir_, epoch=self.epoch_
            )

        else:
            _model_pt = model_pt
            optimizer_pt = model_pt.with_suffix(".optimizer.pt")

        model_state = torch.load(_model_pt, map_location=lambda storage, loc: storage)
        self.model_.load_state_dict(model_state)

        success = self.load_more(model_pt=model_pt)

        if success:

            try:
                optimizer_state = torch.load(
                    optimizer_pt, map_location=lambda storage, loc: storage
                )
                self.optimizer_.load_state_dict(optimizer_state)
            except Exception as e:
                msg = (
                    f"Did not load optimizer state (most likely because current "
                    f"training session uses a different loss than the one used "
                    f"for pre-training)."
                )
                warnings.warn(msg)

        return success

    def load_more(self, model_pt=None) -> bool:
        """Called after model state is loaded

        This method can be overriden to load additional states.
        For instance, it can be used to load the state of a domain classifier
        in domain-adversarial training, or the class centers in center loss.

        Parameters
        ----------
        model_pt : `Path`, optional
            Path to file containing model state.
            Defaults to guessing it from trainer status.

        Returns
        -------
        success : bool
            True if state was loaded successfully, False otherwise.
        """
        return True

    def save_state(self):
        """Save model and optimizer states to disk"""

        # save model state
        model_pt = self.MODEL_PT.format(train_dir=self.train_dir_, epoch=self.epoch_)
        torch.save(self.model_.state_dict(), model_pt)

        # save optimizer state
        optimizer_pt = self.OPTIMIZER_PT.format(
            train_dir=self.train_dir_, epoch=self.epoch_
        )
        torch.save(self.optimizer_.state_dict(), optimizer_pt)

        self.save_more()

    def save_more(self):
        """Called after model and optimizer states are saves

        This method can be overriden to save additional states.
        For instance, it can be used to save the state of a domain classifier
        in domain-adversarial training, or the class centers in center loss.
        """
        pass

    def parameters(self):
        return chain(self.model_.parameters(), self.more_parameters())

    def more_parameters(self):
        """Called by `parameters` method

        This method can be overriden to define additional modules.
        For instance, it can be used to define a domain classifier for
        for domain-adversarial training.

        It should be an iterator yielding the parameters of these additional
        modules.

        Yields
        ------
        parameter : nn.Parameter
            Trainable trainer parameters.
        """
        return []

    def on_train_start(self):
        """Called just before training starts"""
        pass

    def on_epoch_start(self):
        """Called just before epoch starts"""
        pass

    def on_batch_start(self, batch):
        """Called just before batch is processed

        Parameters
        ----------
        batch : `dict`
            Current batch.

        Returns
        -------
        batch : `dict`
            Updated batch.
        """
        return batch

    def on_batch_end(self, loss):
        """Called just after loss is computed

        Parameters
        ----------
        loss : `dict`
            ['loss'] (`torch.Tensor`)
        """
        pass

    def on_epoch_end(self):
        """Called when epoch ends"""
        pass

    def on_train_end(self):
        """Called when training stops"""
        pass

    @property
    def model(self) -> Model:
        return self.model_

    @property
    def optimizer(self):
        return self.optimizer_

    @property
    def specifications(self):
        return self.batch_generator_.specifications

    @property
    def device(self) -> torch.device:
        return self.device_

    @property
    def epoch(self) -> int:
        return self.epoch_

    @property
    def batches_per_epoch(self) -> int:
        return self.batches_per_epoch_

    def get_new_batch(self):
        return next(self.batches_)

    def fit_iter(
        self,
        model: Model,
        batch_generator: BatchGenerator,
        warm_start: Union[int, Path] = 0,
        epochs: int = 1000,
        get_optimizer: Callable[..., Optimizer] = SGD,
        scheduler: Optional[BaseSchedulerCallback] = None,
        learning_rate: Union[Literal["auto"], float] = "auto",
        train_dir: Optional[Path] = None,
        verbosity: int = 2,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
        n_jobs: int = 1,
    ) -> Iterator[Model]:
        """Train model

        Parameters
        ----------
        model : `Model`
            Model.
        batch_generator : `BatchGenerator`
            Batch generator.
        warm_start : `int` or `Path`, optional
            Restart training at this epoch or from this model.
            Default behavior (0) is to train the model from scratch.
        epochs : `int`, optional
            Train model for that many epochs. Defaults to 1000.
        get_optimizer : `callable`, optional
            Callable taking model parameters as input and returns an instance of
            `torch.optim.Optimizer`. May also support `lr` keyword argument.
            Defaults to `torch.optim.SGD`.
        scheduler : `BaseSchedulerCallback`, optional
            Learning rate scheduler. Defaults to `ConstantScheduler`.
        learning_rate : {float, 'auto'}, optional
            Learning rate. Default behavior ('auto') is to use the AutoLR
            heuristic to determine the learning rate automatically.
        train_dir : `Path`, optional
            Directory where models and other log files are stored.
            Defaults to a temporary directory.
        verbosity : `int`, optional
            Set level of verbosity.
            Use 0 (default) to not show any progress bar.
            Use 1 to show a progress bar updated at the end of each epoch.
            Use 2 to add a second progress bar updated at the end of each batch.
        device : `torch.device`, optional
            Device on which the model will be allocated. Defaults to using CPU.
        callbacks : `list` of `Callback` instances
            Add custom callbacks.
        n_jobs : `int`, optional
            Defaults to 1.

        Yields
        ------
        model : `Model`
            Model at current iteration
        """

        #
        if train_dir is None:
            train_dir = Path(tempfile.mkdtemp())
        self.train_dir_ = train_dir

        # DEVICE
        self.device_ = torch.device("cpu") if device is None else device

        # MODEL
        self.model_ = model.to(self.device_)

        # BATCH GENERATOR
        self.batch_generator_ = batch_generator
        self.batches_ = AdaptiveBackgroundGenerator(
            self.batch_generator_, n_jobs=n_jobs
        )
        self.batches_per_epoch_ = self.batch_generator_.batches_per_epoch

        # OPTIMIZER
        lr = ARBITRARY_LR if learning_rate == "auto" else learning_rate
        self.optimizer_ = get_optimizer(self.parameters(), lr=lr)
        self.base_learning_rate_ = learning_rate

        # make sure that 'train_dir' directory does not exist when
        # fine-tuning a pre-trained model or starting from scratch
        # as it might contain the output of very long computations:
        # you do not want to erase them by mistake!

        if isinstance(warm_start, Path) or warm_start == 0:

            try:
                # this will fail if the directory already exists
                os.makedirs(self.train_dir_ / "weights")

            except FileExistsError as e:

                # ask user whether it is OK to continue
                try:
                    with timeout(60):
                        msg = (
                            f'Directory "{self.train_dir_}" exists.\n'
                            f"Are you OK to overwrite existing models? [y/N]: "
                        )
                        overwrite = (input(msg) or "n").lower()
                except TimeoutError:
                    # defaults to "no" after 60 seconds
                    overwrite = "n"

                # stop everything if the user did not say "yes" after a while
                if overwrite != "y":
                    sys.exit()

        # defaults to 0
        self.epoch_ = 0

        # when warm_start is an integer, it means that the user wants to
        # restart training at a given epoch. we intialize the model state and
        # set epoch_ attribute accordingly.
        if isinstance(warm_start, int):

            if warm_start > 0:

                # set epoch_ to requested value...
                self.epoch_ = warm_start

                # ... and load corresponding model if requested
                success = self.load_state(model_pt=None)

        # when warm_start is a Path, it means that the user wants to
        # restart training from a pretrained model
        else:
            try:
                success = self.load_state(model_pt=warm_start)

            except Exception as e:
                msg = (
                    f"Could not assign model weights. The following exception "
                    f"was raised:\n\n{e}\n\nAre you sure the architectures "
                    f"are consistent?"
                )
                sys.exit(msg)

            # save pretrained model as epoch 0
            self.save_state()

        # save specifications to weights/specs.yml
        specs_yml = self.SPECS_YML.format(train_dir=self.train_dir_)
        with io.open(specs_yml, "w") as fp:
            specifications = dict(self.specifications)
            specifications["task"] = str(specifications["task"])
            yaml.dump(specifications, fp, default_flow_style=False)

        # TODO in case success = False, one should freeze the main network for
        # TODO a few epochs and train the "more" part alone first before
        # TODO unfreezing everything. this could be done through a callback

        callbacks_ = []

        # SCHEDULER
        if scheduler is None:
            scheduler = ConstantScheduler()
        callbacks_.append(scheduler)

        logger = Logging(epochs=epochs, verbosity=verbosity)
        callbacks_.append(logger)

        # CUSTOM CALLBACKS
        if callbacks is not None:
            callbacks_.extend(callbacks)

        callbacks = Callbacks(callbacks_)

        # TRAINING STARTS
        self.tensorboard_ = SummaryWriter(
            log_dir=self.train_dir_, purge_step=self.epoch_
        )
        self.on_train_start()
        callbacks.on_train_start(self)

        while self.epoch_ < epochs:

            # EPOCH STARTS
            self.epoch_ += 1
            self.on_epoch_start()
            callbacks.on_epoch_start(self)

            for i in range(self.batches_per_epoch_):

                batch = self.get_new_batch()

                # BATCH IS READY FOR FORWARD PASS
                batch = self.on_batch_start(batch)
                batch = callbacks.on_batch_start(self, batch)

                # FORWARD PASS + LOSS COMPUTATION
                loss = self.batch_loss(batch)

                # BACKWARD PASS
                loss["loss"].backward()
                self.optimizer_.step()
                self.optimizer_.zero_grad()

                # OPTIMIZATION STEP IS DONE
                self.on_batch_end(loss)
                callbacks.on_batch_end(self, loss)

            self.on_epoch_end()
            callbacks.on_epoch_end(self)

            yield self.model_

            self.save_state()

        callbacks.on_train_end(self)

        self.batches_.deactivate()
