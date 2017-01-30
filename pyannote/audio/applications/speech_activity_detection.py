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
import time
import warnings
from os.path import dirname, isfile
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

from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p
import pyannote.core.json

from pyannote.audio.features.utils import Precomputed
import h5py

from .base import Application

import skopt
import skopt.space
from pyannote.metrics.detection import DetectionErrorRate


def tune_binarizer(app, epoch, protocol_name, subset='development'):
    """Tune binarizer

    Parameters
    ----------
    app : SpeechActivityDetection
    epoch : int
        Epoch number.
    protocol_name : str
        E.g. 'Etape.SpeakerDiarization.TV'
    subset : {'train', 'development', 'test'}, optional
        Defaults to 'development'.

    Returns
    -------
    params : dict
        See Binarize.tune
    metric : float
        Best achieved detection error rate
    """

    # initialize protocol
    protocol = get_protocol(protocol_name, progress=False,
                            preprocessors=app.preprocessors_)

    # load model for epoch 'epoch'
    architecture_yml = app.ARCHITECTURE_YML.format(
        train_dir=app.train_dir_)
    weights_h5 = app.WEIGHTS_H5.format(
        train_dir=app.train_dir_, epoch=epoch)
    sequence_labeling = SequenceLabeling.from_disk(
        architecture_yml, weights_h5)

    # initialize sequence labeling
    duration = app.config_['sequences']['duration']
    step = app.config_['sequences']['step']
    aggregation = SequenceLabelingAggregation(
        sequence_labeling, app.feature_extraction_,
        duration=duration, step=step)
    aggregation.cache_preprocessed_ = False

    # tune Binarize thresholds (onset & offset)
    # with respect to detection error rate
    binarize_params, metric = Binarize.tune(
        getattr(protocol, subset)(),
        aggregation.apply,
        get_metric=DetectionErrorRate,
        dimension=1)

    return binarize_params, metric


class SpeechActivityDetection(Application):

    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    TUNE_DIR = '{train_dir}/tune/{protocol}.{subset}'
    APPLY_DIR = '{tune_dir}/apply'

    VALIDATE_TXT = '{train_dir}/validate/{protocol}.{subset}.txt'
    VALIDATE_PNG = '{train_dir}/validate/{protocol}.{subset}.png'

    TUNE_YML = '{tune_dir}/tune.yml'
    TUNE_PNG = '{tune_dir}/tune.png'

    HARD_MDTM = '{apply_dir}/{protocol}.{subset}.mdtm'

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        experiment_dir = dirname(dirname(train_dir))
        speech_activity_detection = cls(experiment_dir, db_yml=db_yml)
        speech_activity_detection.train_dir_ = train_dir
        return speech_activity_detection

    @classmethod
    def from_tune_dir(cls, tune_dir, db_yml=None):
        train_dir = dirname(dirname(tune_dir))
        speech_activity_detection = cls.from_train_dir(train_dir,
                                                       db_yml=db_yml)
        speech_activity_detection.tune_dir_ = tune_dir
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

    def validate(self, protocol_name, subset='development'):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        validation_txt = self.VALIDATE_TXT.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        validation_png = self.VALIDATE_PNG.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(dirname(validation_txt))

        TEMPLATE = '{epoch:04d} {onset:.3f} {offset:.3f} {metric:.6f}\n'
        metrics = []

        with open(validation_txt, mode='w') as fp:

            epoch = 0
            while epoch < 1000:

                # wait until weight file is available
                weights_h5 = self.WEIGHTS_H5.format(
                    train_dir=self.train_dir_,
                    epoch=epoch)
                if not isfile(weights_h5):
                    time.sleep(60)
                    continue

                params, metric = tune_binarizer(
                    self, epoch, protocol_name, subset=subset)

                fp.write(TEMPLATE.format(epoch=epoch, metric=metric, **params))
                fp.flush()

                metrics.append(metric)

                # upldate plot metric = f(epoch)
                best_epoch = np.argmin(metrics)
                best_metric = np.min(metrics)
                fig = plt.figure()
                plt.plot(metrics, 'b')
                plt.plot([best_epoch], [best_metric], 'bo')
                plt.plot([0, epoch], [best_metric, best_metric], 'k--')
                plt.grid(True)
                plt.xlabel('epoch')
                TITLE = '{protocol}.{subset} | DER = {best_metric:.5g} @ #{best_epoch:d}'
                title = TITLE.format(protocol=protocol_name,
                                     subset=subset,
                                     best_metric=best_metric,
                                     best_epoch=best_epoch)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(validation_png, dpi=150)
                plt.close(fig)

    def tune(self, protocol_name, subset='development'):

        tune_dir = self.TUNE_DIR.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(tune_dir)

        epoch = self.get_epochs(self.train_dir_)
        space = [skopt.space.Integer(0, epoch - 1)]

        best_binarize_params = {}
        best_metric = {}

        tune_yml = self.TUNE_YML.format(tune_dir=tune_dir)
        tune_png = self.TUNE_PNG.format(tune_dir=tune_dir)

        def callback(res):

            # plot convergence
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import skopt.plots
            _ = skopt.plots.plot_convergence(res)
            plt.savefig(tune_png, dpi=150)
            plt.close()

            # save state
            params = {'status': {'epochs': epoch,
                                 'objective': float(res.fun)},
                      'epoch': int(res.x[0]),
                      'onset': float(best_binarize_params[tuple(res.x)]['onset']),
                      'offset': float(best_binarize_params[tuple(res.x)]['offset'])
                      }

            with open(tune_yml, 'w') as fp:
                yaml.dump(params, fp, default_flow_style=False)

        def objective_function(params):

            params = tuple(params)
            epoch, = params

            # do not rerun everything if epoch has already been tested
            if params in best_metric:
                return best_metric[params]

            # tune binarizer
            binarize_params, metric = tune_binarizer(
                self, epoch, protocol_name, subset=subset)

            # remember outcome of this trial
            best_binarize_params[params] = binarize_params
            best_metric[params] = metric

            return metric

        res = skopt.gp_minimize(
            objective_function, space, random_state=1337,
            n_calls=20, n_random_starts=10, x0=[epoch - 1],
            verbose=True, callback=callback)

        # TODO tune Binarize a bit longer with the best epoch

        return {'epoch': res.x[0]}, res.fun

    def apply(self, protocol_name, subset='test'):

        apply_dir = self.APPLY_DIR.format(tune_dir=self.tune_dir_)

        mkdir_p(apply_dir)

        # load tuning results
        tune_yml = self.TUNE_YML.format(tune_dir=self.tune_dir_)
        with open(tune_yml, 'r') as fp:
            self.tune_ = yaml.load(fp)

        # load model for epoch 'epoch'
        epoch = self.tune_['epoch']
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

        # initialize protocol
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for i, item in enumerate(getattr(protocol, subset)()):

            prediction = aggregation.apply(item)

            if i == 0:
                # create metadata file at root that contains
                # sliding window and dimension information
                path = Precomputed.get_config_path(apply_dir)
                f = h5py.File(path)
                f.attrs['start'] = prediction.sliding_window.start
                f.attrs['duration'] = prediction.sliding_window.duration
                f.attrs['step'] = prediction.sliding_window.step
                f.attrs['dimension'] = 2
                f.close()

            path = Precomputed.get_path(apply_dir, item)

            # create parent directory
            mkdir_p(dirname(path))

            f = h5py.File(path)
            f.attrs['start'] = prediction.sliding_window.start
            f.attrs['duration'] = prediction.sliding_window.duration
            f.attrs['step'] = prediction.sliding_window.step
            f.attrs['dimension'] = 2
            f.create_dataset('features', data=prediction.data)
            f.close()

        # initialize binarizer
        onset = self.tune_['onset']
        offset = self.tune_['offset']
        binarize = Binarize(onset=onset, offset=offset)

        precomputed = Precomputed(root_dir=apply_dir)

        writer = MDTMParser()
        path = self.HARD_MDTM.format(apply_dir=apply_dir, protocol=protocol_name,
                                subset=subset)
        with open(path, mode='w') as gp:
            for item in getattr(protocol, subset)():
                prediction = precomputed(item)
                segmentation = binarize.apply(prediction, dimension=1)
                writer.write(segmentation.to_annotation(),
                             f=gp, uri=item['uri'], modality='speaker')
