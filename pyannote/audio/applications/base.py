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

import time
import yaml
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
    WEIGHTS_PT = '{train_dir}/weights/{epoch:04d}.pt'

    # created by "validate" mode
    VALIDATE_DIR = '{train_dir}/validate{_task}/{protocol}.{subset}'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None, training=False):
        experiment_dir = dirname(dirname(train_dir))
        app = cls(experiment_dir, db_yml=db_yml, training=training)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_validate_txt(cls, validate_txt, db_yml=None, training=False):
        train_dir = dirname(dirname(dirname(validate_txt)))
        app = cls.from_train_dir(train_dir, db_yml=db_yml, training=training)
        app.validate_txt_ = validate_txt
        return app

    @classmethod
    def from_model_pt(cls, model_pt, db_yml=None, training=False):
        train_dir = dirname(dirname(model_pt))
        app = cls.from_train_dir(train_dir, db_yml=db_yml, training=training)
        app.model_pt_ = model_pt
        epoch = int(basename(app.model_pt_)[:-3])
        app.model_ = app.load_model(epoch, train_dir=train_dir)
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
            self.config_ = yaml.load(fp)

        # preprocessors
        preprocessors = {}
        PREPROCESSORS_DEFAULT = {'audio': db_yml,
                                 'duration': get_audio_duration}

        for key, value in self.config_.get('preprocessors',
                                            PREPROCESSORS_DEFAULT).items():
            if callable(value):
                preprocessors[key] = value
                continue

            try:
                preprocessors[key] = FileFinder(config_yml=value)
            except FileNotFoundError as e:
                preprocessors[key] = value
        self.preprocessors_ = preprocessors

        # scheduler
        SCHEDULER_DEFAULT = {'name': 'DavisKingScheduler',
                             'params': {'learning_rate': 'auto'}}
        scheduler_cfg = self.config_.get('scheduler', SCHEDULER_DEFAULT)
        Scheduler = get_class_by_name(
            scheduler_cfg['name'],
            default_module_name='pyannote.audio.train.schedulers')
        scheduler_params = scheduler_cfg.get('params', {})
        self.get_scheduler_ = partial(Scheduler, **scheduler_params)
        self.learning_rate_ = scheduler_params.get('learning_rate', 'auto')

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

        if training and 'resampling' in self.config_ :
            Resampling = get_class_by_name(
                self.config_['resampling']['name'],
                default_module_name='pyannote.audio.augmentation')
            resampling = Resampling(
                **self.config_['resampling'].get('params', {}))
        else:
            resampling = None

        # feature extraction
        extraction_name = None
        if 'feature_extraction' in self.config_:
            FeatureExtraction = get_class_by_name(
                self.config_['feature_extraction']['name'],
                default_module_name='pyannote.audio.features')
            self.feature_extraction_ = FeatureExtraction(
                **self.config_['feature_extraction'].get('params', {}),
                augmentation=augmentation)


    def train(self, protocol_name, subset='train', restart=None, epochs=1000):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        self.task_.fit(
            self.model_, self.feature_extraction_,
            protocol, subset=subset,
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

        writer = tensorboardX.SummaryWriter(log_dir=str(validate_dir))

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
                best = {'epoch': epoch}
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

                # remember which epoch was processed
                validated_epochs.add(next_epoch_to_validate)

            # increment 'in_order' processing
            if next_epoch_to_validate_in_order == next_epoch_to_validate:
                next_epoch_to_validate_in_order += step
