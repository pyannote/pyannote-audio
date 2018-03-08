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
from os.path import dirname, basename
import numpy as np
from tqdm import tqdm
from glob import glob
from pyannote.database.util import FileFinder
from pyannote.audio.util import mkdir_p
from sortedcontainers import SortedDict
import tensorboardX


class Application(object):

    CONFIG_YML = '{experiment_dir}/config.yml'

    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    WEIGHTS_PT = '{train_dir}/weights/{epoch:04d}.pt'

    # created by "validate" mode
    VALIDATE_DIR = '{train_dir}/validate/{protocol}'
    VALIDATE_TXT = '{validate_dir}/{subset}.{metric}.txt'
    VALIDATE_TXT_TEMPLATE = '{epoch:04d} {value:.6f}\n'
    VALIDATE_PNG = '{validate_dir}/{subset}.{metric}.png'
    VALIDATE_EPS = '{validate_dir}/{subset}.{metric}.eps'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        experiment_dir = dirname(dirname(train_dir))
        app = cls(experiment_dir, db_yml=db_yml)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_validate_txt(cls, validate_txt, db_yml=None):
        train_dir = dirname(dirname(dirname(validate_txt)))
        app = cls.from_train_dir(train_dir, db_yml=db_yml)
        app.validate_txt_ = validate_txt
        return app

    @classmethod
    def from_model_pt(cls, model_pt, db_yml=None):
        train_dir = dirname(dirname(model_pt))
        app = cls.from_train_dir(train_dir, db_yml=db_yml)
        app.model_pt_ = model_pt
        epoch = int(basename(app.model_pt_)[:-3])
        app.model_ = app.load_model(epoch, train_dir=train_dir)
        return app

    def __init__(self, experiment_dir, db_yml=None):
        super(Application, self).__init__()

        self.db_yml = db_yml

        self.preprocessors_ = {'audio': FileFinder(self.db_yml)}

        self.experiment_dir = experiment_dir

        # load configuration
        config_yml = self.CONFIG_YML.format(experiment_dir=self.experiment_dir)
        with open(config_yml, 'r') as fp:
            self.config_ = yaml.load(fp)

        # feature extraction
        if 'feature_extraction' in self.config_:
            extraction_name = self.config_['feature_extraction']['name']
            features = __import__('pyannote.audio.features',
                                  fromlist=[extraction_name])
            FeatureExtraction = getattr(features, extraction_name)
            self.feature_extraction_ = FeatureExtraction(
                **self.config_['feature_extraction'].get('params', {}))

            # do not cache features in memory when they are precomputed on disk
            # as this does not bring any significant speed-up
            # but does consume (potentially) a LOT of memory
            self.cache_preprocessed_ = 'Precomputed' not in extraction_name

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
        self.model_.load_state_dict(torch.load(weights_pt))
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

    def validate_plot(self, metric, values, minimize=True, baseline=None,
                      png=None, eps=None):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # keep track of best epoch so far
        if minimize:
            best_epoch = \
                values.iloc[np.argmin(values.values())]
        else:
            best_epoch = \
                values.iloc[np.argmax(values.values())]

        # corresponding metric value
        best_value = values[best_epoch]

        fig, ax = plt.subplots()
        # plot value = f(epoch)
        ax.plot(values.keys(), values.values(), 'b')
        # mark best epoch
        ax.plot([best_epoch], [best_value], 'bo')
        # horizontal line at best value
        ax.plot([values.iloc[0], values.iloc[-1]],
                [best_value, best_value], 'k--')
        # horizontal line at baseline value
        if baseline is not None:
            ax.plot([values.iloc[0], values.iloc[-1]],
                    [baseline, baseline], 'k', label='baseline')
        ax.grid()
        ax.set_xlabel('epoch')
        ax.set_title('{metric} = {value:.6f} @ epoch #{epoch}'.format(
            metric=metric, value=best_value, epoch=best_epoch))

        fig.tight_layout()

        if png is not None:
            fig.savefig(png, dpi=75)
        if eps is not None:
            fig.savefig(eps)
        plt.close(fig)

        return best_value, best_epoch

    def validate(self, protocol_name, subset='development',
                 every=1, start=0, end=None, in_order=False, **kwargs):

        validate_txt, validate_png, validate_eps = {}, {}, {}
        minimize, baseline, values, best_epoch, best_value = {}, {}, {}, {}, {}

        validate_dir = self.VALIDATE_DIR.format(train_dir=self.train_dir_,
                                                protocol=protocol_name)
        mkdir_p(validate_dir)
        writer = tensorboardX.SummaryWriter(log_dir=validate_dir)

        validation_data = self.validate_init(protocol_name, subset=subset,
                                             **kwargs)

        progress_bar = tqdm(unit='epoch')

        for i, epoch in enumerate(
            self.validate_iter(start=start, end=end, step=every,
                               in_order=in_order)):

            # {'metric1': {'minimize': True, 'value': 0.2, 'baseline': 0.1},
            #  'metric2': {'minimize': False, 'value': 0.9}}
            metrics = self.validate_epoch(epoch, protocol_name, subset=subset,
                                          validation_data=validation_data)

            if i == 0:
                for metric, details in metrics.items():
                    params = {'validate_dir': validate_dir,
                              'subset': subset,
                              'metric': metric}
                    validate_txt[metric] = open(
                        self.VALIDATE_TXT.format(**params), 'w')
                    validate_png[metric] = self.VALIDATE_PNG.format(**params)
                    validate_eps[metric] = self.VALIDATE_EPS.format(**params)
                    minimize[metric] = details.get('minimize', True)
                    baseline[metric] = details.get('baseline', None)
                    values[metric] = SortedDict()

            description = 'Epoch #{epoch}'.format(epoch=epoch)

            for metric, details in sorted(metrics.items()):
                value = details['value']
                values[metric][epoch] = value

                # save metric value to file
                line = self.VALIDATE_TXT_TEMPLATE.format(
                    epoch=epoch, value=values[metric][epoch])
                validate_txt[metric].write(line)
                validate_txt[metric].flush()

                writer.add_scalar(
                    metric, values[metric][epoch], global_step=epoch)

                best_value, best_epoch = self.validate_plot(
                    metric, values[metric],
                    minimize=minimize[metric],
                    baseline=baseline[metric],
                    png=validate_png[metric],
                    eps=validate_eps[metric])

                if abs(best_value) < 1:
                    addon = (' : {metric} = {value:.3f}% '
                             '[{best_value:.3f}%, #{best_epoch}]')
                    description += addon.format(metric=metric, value=100 * value,
                                                best_value=100 * best_value,
                                                best_epoch=best_epoch)
                else:
                    addon = (' : {metric} = {value:.3f} '
                             '[{best_value:.3f}, #{best_epoch}]')
                    description += addon.format(metric=metric, value=value,
                                                best_value=best_value,
                                                best_epoch=best_epoch)

            progress_bar.set_description(description)
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
            # always process it first
            if (not in_order) and (last_completed_epoch not in validated_epochs):
                next_epoch_to_validate = last_completed_epoch
                time.sleep(5)  # HACK give checkpoint time to save weights

            # in case no new epoch has completed since last time
            # process the next epoch in chronological order (if available)
            elif next_epoch_to_validate_in_order < last_completed_epoch:
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
