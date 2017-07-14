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


import yaml
import os.path
from glob import glob
from pyannote.database.util import FileFinder


class Application(object):

    CONFIG_YML = '{experiment_dir}/config.yml'
    WEIGHTS_H5 = '{train_dir}/weights/{epoch:04d}.h5'

    def __init__(self, experiment_dir, db_yml=None):
        super(Application, self).__init__()

        self.db_yml = db_yml
        self.preprocessors_ = {'wav': FileFinder(self.db_yml)}

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

        directory = self.WEIGHTS_H5.format(train_dir=train_dir, epoch=0)[:-7]
        weights_h5 = glob(directory + '*[0-9][0-9][0-9][0-9].h5')

        if not weights_h5:
            number_of_epochs = 0
            first_epoch = None

        else:
            number_of_epochs = int(os.path.basename(weights_h5[-1])[:-3]) + 1
            first_epoch = int(os.path.basename(weights_h5[0])[:-3])

        return number_of_epochs, first_epoch if return_first \
                                             else number_of_epochs

    def epoch_iter(self, start=0, step=1, sleep=60):
        """Usage:

        >>> # initialize epoch generator
        >>> for epoch in app.epoch_iter():
        ...     validate(epoch)
        """

        processed_epochs = set()
        next_epoch_to_process_in_order = start

        # wait for first epoch to complete
        while True:

            _, first_epoch = self.get_number_of_epochs(return_first=True)
            if first_epoch is None:
                time.sleep(sleep)
                continue

            # corner case: make sure this does not wait forever
            # for epoch 'start' as it might never happen, in case
            # training is started after n pre-existing epochs
            if next_epoch_to_process_in_order < first_epoch:
                next_epoch_to_process_in_order = first_epoch

            # first epoch has completed
            break

        while True:

            # check last completed epoch
            last_completed_epoch = self.get_number_of_epochs() - 1

            # if last completed epoch has not been processed yet,
            # always process it first
            if last_completed_epoch not in processed_epochs:
                next_epoch_to_process = last_completed_epoch

            # in case no new epoch has completed since last time
            # process the next epoch in chronological order (if available)
            elif next_epoch_to_process_in_order < last_completed_epoch:
                next_epoch_to_process = next_epoch_to_process_in_order

            #  otherwise, just wait for a new epoch to complete
            else:
                time.sleep(sleep)
                continue

            # yield next epoch to process
            yield next_epoch_to_process

            # remember which epoch was processed
            processed_epochs.add(next_epoch_to_process)

            # increment 'in_order' processing
            if next_epoch_to_process_in_order == next_epoch_to_process:
                next_epoch_to_process_in_order += step
