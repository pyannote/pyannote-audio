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

from pyannote.database.util import FileFinder


class Application(object):

    CONFIG_YML = '{experiment_dir}/config.yml'
    ARCHITECTURE_YML = '{train_dir}/architecture.yml'
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
        feature_extraction_name = self.config_['feature_extraction']['name']
        features = __import__('pyannote.audio.features',
                              fromlist=[feature_extraction_name])
        FeatureExtraction = getattr(features, feature_extraction_name)
        self.feature_extraction_ = FeatureExtraction(
            **self.config_['feature_extraction'].get('params', {}))

        # do not cache features in memory when they are precomputed on disk
        # as this does not bring any significant speed-up
        # but does consume (potentially) a LOT of memory
        self.cache_preprocessed_ = 'Precomputed' not in feature_extraction_name

    def get_epochs(self, train_dir):
        """Get current number of completed epochs"""

        epoch = 0

        while True:
            weights_h5 = self.WEIGHTS_H5.format(train_dir=train_dir, epoch=epoch)
            if not os.path.isfile(weights_h5):
                break
            epoch += 1

        return epoch
