#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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
import os
import sys
import time
import yaml
from typing import Optional
from pathlib import Path
from os.path import dirname, basename
import numpy as np
from tqdm import tqdm
from glob import glob
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.util import mkdir_p
from pyannote.audio.features.utils import get_audio_duration
from sortedcontainers import SortedDict
import tensorboardX
from functools import partial
from pyannote.core.utils.helper import get_class_by_name
import warnings


class Application(object):

    CONFIG_YML = '{experiment_dir}/config.yml'
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    WEIGHTS_DIR = '{train_dir}/weights'
    WEIGHTS_PT = '{train_dir}/weights/{epoch:04d}.pt'
    VALIDATE_DIR = '{train_dir}/validate{_task}/{protocol}.{subset}'
    APPLY_DIR = '{validate_dir}/apply/{epoch:04d}'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None, training=False):
        experiment_dir = dirname(dirname(train_dir))
        app = cls(experiment_dir, db_yml=db_yml, training=training)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_model_pt(cls, model_pt, db_yml=None, training=False):
        train_dir = dirname(dirname(model_pt))
        app = cls.from_train_dir(train_dir, db_yml=db_yml, training=training)
        app.model_pt_ = model_pt
        epoch = int(basename(app.model_pt_)[:-3])
        app.model_ = app.load_model(epoch, train_dir=train_dir)
        app.epoch_ = epoch
        return app

    @classmethod
    def from_validate_dir(cls, validate_dir: Path,
                               db_yml: Optional[Path] = None,
                               training: Optional[bool] = False):

        # infer train directory from validate directory
        train_dir = dirname(dirname(validate_dir))

        # load params.yml file from validate directory
        with open(validate_dir / 'params.yml', 'r') as fp:
            params_yml = yaml.load(fp)

        # build path to best epoch model
        epoch = params_yml['epoch']
        model_pt = self.WEIGHTS_PT.format(train_dir=train_dir,
                                          epoch=epoch)

        # instantiate application
        # TODO. get rid of from_model_pt
        app = cls.from_model_pt(model_t, db_yml=db_yml, training=training)
        app.validate_dir_ = validate_dir
        app.epoch_ = epoch

        # keep track of pipeline parameters
        app.pipeline_params_ = params_yml.get('params', {})

        return app

    def __init__(self, experiment_dir, db_yml=None, training=False):
        """

        Parameters
        ----------
        experiment_dir : str
        db_yml : str, optional
        training : boolean, optional
            When False, data augmentation is disabled.
        """
        super(Application, self).__init__()

        self.experiment_dir = experiment_dir

        # load configuration
        config_yml = self.CONFIG_YML.format(experiment_dir=self.experiment_dir)
        with open(config_yml, 'r') as fp:
            self.config_ = yaml.load(fp, Loader=yaml.SafeLoader)

        # preprocessors
        preprocessors = {'audio': FileFinder(db_yml),
                         'duration': get_audio_duration}

        for key, preprocessor in self.config_.get('preprocessors', {}).items():
            # preprocessors:
            #    key:
            #       name: package.module.ClassName
            #       params:
            #          param1: value1
            #          param2: value2
            if isinstance(preprocessor, dict):
                Klass = get_class_by_name(preprocessor['name'])
                preprocessors[key] = Klass(**preprocessor.get('params', {}))
                continue

            try:
                # preprocessors:
                #    key: /path/to/database.yml
                preprocessors[key] = FileFinder(preprocessor)

            except FileNotFoundError as e:
                # preprocessors:
                #    key: /path/to/{uri}.wav
                preprocessors[key] = preprocessor

        self.preprocessors_ = preprocessors

        # scheduler
        SCHEDULER_DEFAULT = {'name': 'DavisKingScheduler',
                             'params': {'learning_rate': 'auto'}}
        scheduler_cfg = self.config_.get('scheduler', SCHEDULER_DEFAULT)
        Scheduler = get_class_by_name(
            scheduler_cfg['name'],
            default_module_name='pyannote.audio.train.schedulers')
        scheduler_params = scheduler_cfg.get('params', {})
        self.learning_rate_ = scheduler_params.pop('learning_rate', 'auto')
        self.get_scheduler_ = partial(Scheduler, **scheduler_params)

        # optimizer
        OPTIMIZER_DEFAULT = {
            'name': 'SGD',
            'params': {'momentum': 0.9, 'dampening': 0, 'weight_decay': 0,
                       'nesterov': True}}
        optimizer_cfg = self.config_.get('optimizer', OPTIMIZER_DEFAULT)
        try:
            Optimizer = get_class_by_name(optimizer_cfg['name'],
                                          default_module_name='torch.optim')
            optimizer_params = optimizer_cfg.get('params', {})
            self.get_optimizer_ = partial(Optimizer, **optimizer_params)

        # do not raise an error here as it is possible that the optimizer is
        # not really needed (e.g. in pipeline training)
        except ModuleNotFoundError as e:
            warnings.warn(e.args[0])

        # data augmentation (only when training the model)
        if training and 'data_augmentation' in self.config_ :
            DataAugmentation = get_class_by_name(
                self.config_['data_augmentation']['name'],
                default_module_name='pyannote.audio.augmentation')
            augmentation = DataAugmentation(
                **self.config_['data_augmentation'].get('params', {}))
        else:
            augmentation = None

        # feature extraction
        if 'feature_extraction' in self.config_:
            FeatureExtraction = get_class_by_name(
                self.config_['feature_extraction']['name'],
                default_module_name='pyannote.audio.features')
            self.feature_extraction_ = FeatureExtraction(
                **self.config_['feature_extraction'].get('params', {}),
                augmentation=augmentation)

    def train(self, protocol_name, subset='train', restart=0, epochs=1000):
        """Trainer model

        Parameters
        ----------
        protocol_name : `str`
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'train'.
        restart : `int`, optional
            Restart training at `restart`th epoch. Defaults to training from
            scratch.
        epochs : `int`, optional
            Train for that many epochs. Defaults to 1000.
        """

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        if not restart:

            weights_dir = self.task_.WEIGHTS_DIR.format(log_dir=train_dir)
            try:
                # this will fail if the directory already exists
                # and this is OK  because 'weights' directory
                # usually contains the output of very long computations
                # and you do not want to erase them by mistake :/
                os.makedirs(weights_dir)
            except FileExistsError as e:
                msg = (
                    f'You are about to overwrite pretrained models in '
                    f'"{weights_dir}" directory. If you want to train a new '
                    f'model from scratch, first (backup and) remove the '
                    f'directory.'
                )
                sys.exit(msg)

        # initialize batch generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)
        batch_generator = self.task_.get_batch_generator(
            self.feature_extraction_, protocol, subset=subset,
            frame_info=self.frame_info_, frame_crop=self.frame_crop_)

        self.task_.fit(
            self.get_model_, batch_generator,
            restart=restart, epochs=epochs,
            get_optimizer=self.get_optimizer_,
            get_scheduler=self.get_scheduler_,
            learning_rate=self.learning_rate_,
            log_dir=train_dir, device=self.device)

    def load_model(self, epoch, train_dir=None):
        """Load pretrained model

        Parameters
        ----------
        epoch : int
            Which epoch to load.
        train_dir : str, optional
            Path to train directory. Defaults to self.train_dir_.
        """

        if train_dir is None:
            train_dir = self.train_dir_

        # initialize model from specs stored on disk
        specs_yml = self.task_.SPECS_YML.format(log_dir=train_dir)
        with io.open(specs_yml, 'r') as fp:
            specifications = yaml.load(fp, Loader=yaml.SafeLoader)
        self.model_ = self.get_model_(specifications)

        import torch
        weights_pt = self.WEIGHTS_PT.format(
            train_dir=train_dir, epoch=epoch)

        # if GPU is not available, load using CPU
        self.model_.load_state_dict(
            torch.load(weights_pt, map_location=lambda storage, loc: storage))

        return self.model_

    def get_number_of_epochs(self, train_dir=None, return_first=False):
        """Get information about completed epochs

        Parameters
        ----------
        train_dir : str, optional
            Training directory. Defaults to self.train_dir_
        return_first : bool, optional
            Defaults (False) to return number of epochs.
            Set to True to also return index of first epoch.

        """

        if train_dir is None:
            train_dir = self.train_dir_

        directory = self.WEIGHTS_PT.format(train_dir=train_dir, epoch=0)[:-7]
        weights = sorted(glob(directory + '*[0-9][0-9][0-9][0-9].pt'))

        if not weights:
            number_of_epochs = 0
            first_epoch = None

        else:
            number_of_epochs = int(basename(weights[-1])[:-3]) + 1
            first_epoch = int(basename(weights[0])[:-3])

        return (number_of_epochs, first_epoch) if return_first \
                                               else number_of_epochs

    def validate_init(self, protocol_name, subset='development'):
        pass

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):
        raise NotImplementedError('')

    def validate(self, protocol_name, subset='development',
                 every=1, start=0, end=None, in_order=False, task=None, **kwargs):

        validate_dir = Path(self.VALIDATE_DIR.format(
            train_dir=self.train_dir_,
            _task=f'_{task}' if task is not None else '',
            protocol=protocol_name, subset=subset))

        params_yml = validate_dir / 'params.yml'
        validate_dir.mkdir(parents=True, exist_ok=False)

        writer = tensorboardX.SummaryWriter(logdir=str(validate_dir))

        validation_data = self.validate_init(protocol_name, subset=subset,
                                             **kwargs)

        progress_bar = tqdm(unit='iteration')

        for i, epoch in enumerate(
            self.validate_iter(start=start, end=end, step=every,
                               in_order=in_order)):

            # {'metric': 'detection_error_rate',
            #  'minimize': True,
            #  'value': 0.9,
            #  'pipeline': ...}
            details = self.validate_epoch(
                epoch, protocol_name, subset=subset,
                validation_data=validation_data)

            # initialize
            if i == 0:
                # what is the name of the metric?
                metric = details['metric']
                # should the metric be minimized?
                minimize = details['minimize']
                # epoch -> value dictionary
                values = SortedDict()

            # metric value for current epoch
            values[epoch] = details['value']

            # send value to tensorboard
            writer.add_scalar(
                f'validate/{protocol_name}.{subset}/{metric}',
                values[epoch], global_step=epoch)

            # keep track of best value so far
            if minimize:
                best_epoch = values.iloc[np.argmin(values.values())]
                best_value = values[best_epoch]

            else:
                best_epoch = values.iloc[np.argmax(values.values())]
                best_value = values[best_epoch]

            # if current epoch leads to the best metric so far
            # store both epoch number and best pipeline parameter to disk
            if best_epoch == epoch:
                best = {
                    metric: best_value,
                    'epoch': epoch,
                }
                if 'pipeline' in details:
                    pipeline = details['pipeline']
                    best['params'] = pipeline.parameters(instantiated=True)
                with open(params_yml, mode='w') as fp:
                    fp.write(yaml.dump(best, default_flow_style=False))

            # progress bar
            desc = (f'{metric} | '
                    f'Epoch #{best_epoch} = {100 * best_value:g}% (best) | '
                    f'Epoch #{epoch} = {100 * details["value"]:g}%')
            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

    def validate_iter(self, start=None, end=None, step=1, sleep=10,
                      in_order=False):
        """Continuously watches `train_dir` for newly completed epochs
        and yields them for validation

        Note that epochs will not necessarily be yielded in order.
        The very last completed epoch will always be first on the list.

        Parameters
        ----------
        start : int, optional
            Start validating after `start` epochs. Defaults to 0.
        end : int, optional
            Stop validating after epoch `end`. Defaults to never stop.
        step : int, optional
            Validate every `step`th epoch. Defaults to 1.
        sleep : int, optional
        in_order : bool, optional
            Force chronological validation.

        Usage
        -----
        >>> for epoch in app.validate_iter():
        ...     app.validate(epoch)


        """

        if end is None:
            end = np.inf

        if start is None:
            start = 0

        validated_epochs = set()
        next_epoch_to_validate_in_order = start

        while next_epoch_to_validate_in_order < end:

            # wait for first epoch to complete
            _, first_epoch = self.get_number_of_epochs(return_first=True)
            if first_epoch is None:
                print('waiting for first epoch to complete...')
                time.sleep(sleep)
                continue

            # corner case: make sure this does not wait forever
            # for epoch 'start' as it might never happen, in case
            # training is started after n pre-existing epochs
            if next_epoch_to_validate_in_order < first_epoch:
                next_epoch_to_validate_in_order = first_epoch

            # first epoch has completed
            break

        while True:

            # check last completed epoch
            last_completed_epoch = self.get_number_of_epochs() - 1

            # if last completed epoch has not been processed yet,
            # always process it first (except if 'in order')
            if (not in_order) and (last_completed_epoch not in validated_epochs):
                next_epoch_to_validate = last_completed_epoch
                time.sleep(5)  # HACK give checkpoint time to save weights

            # in case no new epoch has completed since last time
            # process the next epoch in chronological order (if available)
            elif next_epoch_to_validate_in_order <= last_completed_epoch:
                next_epoch_to_validate = next_epoch_to_validate_in_order

            # otherwise, just wait for a new epoch to complete
            else:
                time.sleep(sleep)
                continue

            if next_epoch_to_validate not in validated_epochs:

                # yield next epoch to process
                yield next_epoch_to_validate

                # stop validation when the last epoch has been reached
                if next_epoch_to_validate >= end:
                    return

                # remember which epoch was processed
                validated_epochs.add(next_epoch_to_validate)

            # increment 'in_order' processing
            if next_epoch_to_validate_in_order == next_epoch_to_validate:
                next_epoch_to_validate_in_order += step
