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

"""
Speech activity detection

Usage:
  pyannote-speech-detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  pyannote-speech-detection validate [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
  pyannote-speech-detection tune  [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
  pyannote-speech-detection apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  pyannote-speech-detection -h | --help
  pyannote-speech-detection --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <train_dir>                Set path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  <tune_dir>                 Set path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default subset is "train".
                             In "tune" mode, default subset is "development".
                             In "apply" mode, default subset is "test".
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence labeling (0 vs. 1, non-speech vs. speech), the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    both training and testing.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives

    architecture:
       name: StackedLSTM
       params:                       # this experiments relies
         n_classes: 2                # on one LSTM layer (16 outputs)
         lstm: [16]                  # and one dense layer.
         mlp: [16]                   # LSTM is bidirectional
         bidirectional: ave

    sequences:
       duration: 3.2                 # this experiments relies
       step: 0.8                     # on sliding windows of 3.2s
                                     # with a step of 0.8s
       batch_size: 1024
    ...................................................................

"train" mode:
    First, one should train the raw sequence labeling neural network using
    "train" mode. This will create the following directory that contains
    the pre-trained neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "tune" mode.

"tune" mode:
    Then, one should tune the hyper-parameters using "tune" mode.
    This will create the following files describing the best hyper-parameters
    to use:

        <train_dir>/tune/<database.task.protocol>.<subset>/tune.yml
        <train_dir>/tune/<database.task.protocol>.<subset>/tune.png

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequent "apply" mode.

"apply" mode
    Finally, one can apply speech activity detection using "apply" mode.
    This will create the following files that contains the hard (mdtm) and
    soft (h5) outputs of speech activity detection.

        <tune_dir>/apply/<database.task.protocol>.<subset>.mdtm
        <tune_dir>/apply/{database}/{uri}.h5

    This means that file whose unique resource identifier is {uri} has been
    processed.

"""

import io
import yaml
import time
import warnings
from os.path import dirname, isfile, expanduser
import numpy as np
from collections import Counter

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.speech import \
    SpeechActivityDetectionBatchGenerator
from pyannote.audio.optimizers import SSMORMS3

from pyannote.audio.callback import LoggingCallback

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Binarize
from pyannote.database.util import get_unique_identifier
from pyannote.database.util import get_annotated
from pyannote.database import get_protocol

from pyannote.metrics.binary_classification import det_curve

from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p
import pyannote.core.json

from pyannote.audio.features.utils import Precomputed
import h5py

from .base import Application

from tqdm import tqdm

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
    sequence_labeling = SequenceLabeling.from_disk(
        app.train_dir_, epoch)

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

    # created by "train" mode
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    APPLY_DIR = '{tune_dir}/apply'

    # created by "validate" mode
    VALIDATE_DIR = '{train_dir}/validate/{protocol}'
    VALIDATE_TXT = '{validate_dir}/{subset}.eer.txt'
    VALIDATE_TXT_TEMPLATE = '{epoch:04d} {eer:5f}\n'
    VALIDATE_PNG = '{validate_dir}/{subset}.eer.png'
    VALIDATE_EPS = '{validate_dir}/{subset}.eer.eps'

    # created by "tune" mode
    TUNE_DIR = '{train_dir}/tune/{protocol}.{subset}'
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
        batch_size = self.config_['sequences'].get('batch_size', 8192)
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
        # number of batches per epoch
        steps_per_epoch = int(np.ceil((train_total / step) / batch_size))

        # input shape (n_frames, n_features)
        input_shape = batch_generator.shape

        # generator that loops infinitely over all training files
        train_files = getattr(protocol, subset)()
        generator = batch_generator(train_files, infinite=True)

        labeling = SequenceLabeling()
        labeling.fit(input_shape, self.architecture_,
                     generator, steps_per_epoch, 1000,
                     optimizer=SSMORMS3(), log_dir=train_dir)

        return labeling

    def _validation_set(self, protocol_name, subset='development'):
        # this generator is hacked to generate y_true
        # (which is stored in its internal preprocessed_ attribute)
        batch_generator = SpeechActivityDetectionBatchGenerator(
            self.feature_extraction_)
        batch_generator.cache_preprocessed_ = True

        # iterate over each test file and generate y_true
        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        file_generator = getattr(protocol, subset)()
        for current_file in file_generator:
            identifier = get_unique_identifier(current_file)
            batch_generator.preprocess(current_file, identifier=identifier)

        return batch_generator.preprocessed_['y']

    def validate(self, protocol_name, subset='development'):

        # prepare paths
        validate_dir = self.VALIDATE_DIR.format(train_dir=self.train_dir_,
                                                protocol=protocol_name)
        validate_txt = self.VALIDATE_TXT.format(validate_dir=validate_dir,
                                                subset=subset)
        validate_png = self.VALIDATE_PNG.format(validate_dir=validate_dir,
                                                subset=subset)
        validate_eps = self.VALIDATE_EPS.format(validate_dir=validate_dir,
                                                subset=subset)

        # create validation directory
        mkdir_p(validate_dir)

        # Build validation set
        y = self._validation_set(protocol_name, subset=subset)

        # list of equal error rates, and current epoch
        eers, epoch = [], 0

        desc_format = ('EER = {eer:.2f}% @ epoch #{epoch:d} ::'
                      ' Best EER = {best_eer:.2f}% @ epoch #{best_epoch:d} :')
        progress_bar = tqdm(unit='epoch', total=1000)

        with open(validate_txt, mode='w') as fp:

            # watch and evaluate forever
            while True:

                weights_h5 = LoggingCallback.WEIGHTS_H5.format(
                    log_dir=self.train_dir_, epoch=epoch)

                # wait until weight file is available
                if not isfile(weights_h5):
                    time.sleep(60)
                    continue

                # load model for current epoch
                sequence_labeling = SequenceLabeling.from_disk(
                    self.train_dir_, epoch)

                # initialize sequence labeling
                duration = self.config_['sequences']['duration']
                step = duration   # hack to make things faster
                # step = self.config_['sequences']['step']
                aggregation = SequenceLabelingAggregation(
                    sequence_labeling, self.feature_extraction_,
                    duration=duration, step=step)
                aggregation.cache_preprocessed_ = False

                # estimate equal error rate (average of all files)
                eers_ = []
                protocol = get_protocol(protocol_name, progress=False,
                                        preprocessors=self.preprocessors_)
                file_generator = getattr(protocol, subset)()
                for current_file in file_generator:
                    identifier = get_unique_identifier(current_file)
                    uem = get_annotated(current_file)
                    y_true = y[identifier].crop(uem)[:, 1]
                    counts = Counter(y_true)
                    if counts[0] * counts[1] == 0:
                        continue
                    y_pred = aggregation.apply(current_file).crop(uem)[:, 1]

                    _, _, _, eer = det_curve(y_true, y_pred, distances=False)

                    eers_.append(eer)
                eer = np.mean(eers_)
                eers.append(eer)

                # save equal error rate to file
                fp.write(self.VALIDATE_TXT_TEMPLATE.format(
                    epoch=epoch, eer=eer))
                fp.flush()

                # keep track of best epoch so far
                best_epoch, best_eer = np.argmin(eers), np.min(eers)

                progress_bar.set_description(
                    desc_format.format(epoch=epoch, eer=100*eer,
                                       best_epoch=best_epoch,
                                       best_eer=100*best_eer))
                progress_bar.update(1)

                # plot
                fig = plt.figure()
                plt.plot(eers, 'b')
                plt.plot([best_epoch], [best_eer], 'bo')
                plt.plot([0, epoch], [best_eer, best_eer], 'k--')
                plt.grid(True)
                plt.xlabel('epoch')
                plt.ylabel('EER on {subset}'.format(subset=subset))
                TITLE = '{best_eer:.5g} @ epoch #{best_epoch:d}'
                title = TITLE.format(best_eer=best_eer,
                                     best_epoch=best_epoch,
                                     subset=subset)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(validate_png, dpi=75)
                plt.savefig(validate_eps)
                plt.close(fig)

                # validate next epoch
                epoch += 1

        progress_bar.close()

    def tune(self, protocol_name, subset='development'):

        tune_dir = self.TUNE_DIR.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(tune_dir)

        epoch, first_epoch = self.get_number_of_epochs(self.train_dir_, return_first=True)
        space = [skopt.space.Integer(first_epoch, epoch - 1)]

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
            plt.savefig(tune_png, dpi=75)
            plt.close()

            # save state
            params = {'status': {'epochs': epoch,
                                 'objective': float(res.fun)},
                      'epoch': int(res.x[0]),
                      'onset': float(best_binarize_params[tuple(res.x)]['onset']),
                      'offset': float(best_binarize_params[tuple(res.x)]['offset'])
                      }

            with io.open(tune_yml, 'w') as fp:
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
        with io.open(tune_yml, 'r') as fp:
            self.tune_ = yaml.load(fp)

        # load model for epoch 'epoch'
        epoch = self.tune_['epoch']
        sequence_labeling = SequenceLabeling.from_disk(
            self.train_dir_, epoch)

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
        with io.open(path, mode='w') as gp:
            for item in getattr(protocol, subset)():
                prediction = precomputed(item)
                segmentation = binarize.apply(prediction, dimension=1)
                writer.write(segmentation.to_annotation(),
                             f=gp, uri=item['uri'], modality='speaker')


def main():

    arguments = docopt(__doc__, version='Speech activity detection')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'train'
        application = SpeechActivityDetection(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset)

    if arguments['validate']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.validate(protocol_name, subset=subset)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.tune(protocol_name, subset=subset)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']
        if subset is None:
            subset = 'test'
        application = SpeechActivityDetection.from_tune_dir(
            tune_dir, db_yml=db_yml)
        application.apply(protocol_name, subset=subset)
