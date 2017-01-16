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


import os.path
import numpy as np

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.speech import \
    SpeechActivityDetectionBatchGenerator
from pyannote.audio.optimizers import SSMORMS3

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Binarize
from pyannote.database.util import get_unique_identifier
from pyannote.database.util import get_annotated

from .base import Application

import skopt
import skopt.space
from pyannote.metrics.detection import DetectionErrorRate


class SpeechActivityDetection(Application):

    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    TUNE_DIR = '{train_dir}/tune/{protocol}.{subset}'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        experiment_dir = os.path.dirname(os.path.dirname(train_dir))
        return cls.__init__(experiment_dir, db_yml=db_yml)

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeechActivityDetection, self).__init__(
            experiment_dir, db_yml=db_yml)

        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.labeling.models',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        self.architecture_ = Architecture(
            **self.config_['architecture'].get('params', {}))

    def train(self, protocol_name, subset='train'):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        # sequence batch generator
        batch_size = 8192
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        batch_generator = SpeechActivityDetectionBatchGenerator(
            self.feature_extraction_, duration=duration, step=step,
            batch_size=batch_size)
        batch_generator.cache_preprocessed_ = self.cache_preprocessed_

        protocol = self.get_protocol(protocol_name, progress=False)

        # total train duration
        train_total = protocol.stats(subset)['annotated']
        # number of samples per epoch + round it to closest batch
        samples_per_epoch = batch_size * \
            int(np.ceil((train_total / step) / batch_size))

        # input shape (n_frames, n_features)
        input_shape = batch_generator.shape

        # generator that loops infinitely over all training files
        train_files = getattr(protocol, subset)()
        generator = batch_generator(train_files, infinite=True)

        labeling = SequenceLabeling()
        labeling.fit(input_shape, self.architecture_,
                     generator, samples_per_epoch, 1000,
                     optimizer=SSMORMS3(), log_dir=train_dir)

        return labeling

    def tune(self, train_dir, protocol_name, subset='development'):

        tune_dir = self.TUNE_DIR.format(
            train_dir=train_dir,
            protocol=protocol_name,
            subset=subset)

        epoch = self.get_epoch(train_dir)
        space = [skopt.space.Integer(0, epoch - 1)]

        best_params = {}

        def objective_function(params):

            epoch, = params

            # load model obtained after that many training epochs
            architecture_yml = self.ARCHITECTURE_YML.format(
                train_dir=train_dir)
            weights_h5 = self.WEIGHTS_H5.format(
                train_dir=train_dir, epoch=epoch)
            sequence_labeling = SequenceLabeling.from_disk(
                architecture_yml, weights_h5)

            # process each development file with that model
            # `predictions[uri]` contains soft sequence labeling
            aggregation = SequenceLabelingAggregation(
                sequence_labeling, feature_extraction,
                duration=duration, step=step)
            protocol = self.get_protocol(protocol_name, progress=False)
            predictions = {get_unique_identifier(item): aggregation.apply(item)
                           for item in getattr(protocol, subset)()}

            # tune Binarize thresholds (onset & offset)
            # with respect to detection error rate
            params, metric = Binarize.tune(
                predictions, protocol_name, subset=subset, n_calls=20,
                get_metric=DetectionErrorRate, returns_metric=True)

            best_params[epoch] = params

            print(epoch, metric, params)

            # TODO store every trial

            return metric

        res = skopt.gp_minimize(
            objective_function, space,
            n_calls=3, x0=[epoch - 1], verbose=True,
            n_random_starts=5, random_state=1337)

        best_epoch = res.x[0]
        print(best_params[best_epoch])
        print(res.fun)
