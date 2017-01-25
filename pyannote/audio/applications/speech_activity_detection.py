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
from pyannote.database import get_protocol

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
        speech_activity_detection = cls(experiment_dir, db_yml=db_yml)
        speech_activity_detection.train_dir_ = train_dir
        return speech_activity_detection

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

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

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

    def tune(self, protocol_name, subset='development'):

        tune_dir = self.TUNE_DIR.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        epoch = self.get_epochs(self.train_dir_)
        space = [skopt.space.Integer(0, epoch - 1)]

        best_binarize_params = {}
        best_metric = {}

        def callback(res):

            # TODO add pretty convergence plots...

            params = {'status': {'epochs': epoch, 'objective': res.fun},
                      'epoch': int(res.x[0]),
                      'onset': float(best_binarize_params[res.x][0]),
                      'offset': float(best_binarize_params[res.x][1])
                      }

            with open(tune_dir + '/tune.yml', 'w') as fp:
                yaml.dump(params, fp, default_flow_style=False)

        def objective_function(params):

            epoch, = params

            # do not rerun everything if epoch has already been tested
            if params in best_metric:
                return best_metric[params]

            # initialize protocol
            protocol = get_protocol(protocol_name, progress=False,
                                    preprocessors=self.preprocessors_)

            # load model for epoch 'epoch'
            architecture_yml = self.ARCHITECTURE_YML.format(
                train_dir=self.train_dir_)
            weights_h5 = self.WEIGHTS_H5.format(
                train_dir=self.train_dir_, epoch=epoch)
            sequence_labeling = SequenceLabeling.from_disk(
                architecture_yml, weights_h5)

            # initialize sequence labeling
            duration = self.config_['sequences']['duration']
            step = self.config_['sequences']['step']
            aggregation = SequenceLabelingAggregation(
                sequence_labeling, self.feature_extraction_,
                duration=duration, step=step)

            # tune Binarize thresholds (onset & offset)
            # with respect to detection error rate
            binarize_params, metric = Binarize.tune(
                getattr(protocol, subset)(),
                aggregation.apply,
                get_metric=DetectionErrorRate,
                dimension=1)

            # remember outcome of this trial
            best_binarize_params[params] = binarize_params
            best_metric[params] = metric

            return metric

        res = skopt.gp_minimize(
            objective_function, space, random_state=1337,
            n_calls=20, n_random_starts=10, x0=[epoch - 1],
            verbose=True, callback=callback)

        tune_dir + '/tune.yml'

        return {'epoch': res.x[0]}, res.fun
