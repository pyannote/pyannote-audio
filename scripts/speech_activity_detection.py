#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
  speech_activity_detection train [--subset=<subset>] <experiment_dir> <database.task.protocol> <wav_template>
  speech_activity_detection tune  [--subset=<subset> --recall=<beta>] <train_dir> <database.task.protocol> <wav_template>
  speech_activity_detection apply [--subset=<subset> --recall=<beta>] <tune_dir> <database.task.protocol> <wav_template>
  speech_activity_detection -h | --help
  speech_activity_detection --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <wav_template>             Set path to actual wave files. This path is
                             expected to contain a {uri} placeholder that will
                             be replaced automatically by the actual unique
                             resource identifier (e.g. '/Etape/{uri}.wav').
  <train_dir>                Set path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  <tune_dir>                 Set path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default subset is "train".
                             In "tune" mode, default subset is "development".
                             In "apply" mode, default subset is "test".
  --recall=<beta>            Set importance of recall with respect to precision.
                             [default: 1.0]
                             Use higher values if you want to improve recall.
  -h --help                  Show this screen.
  --version                  Show version.

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
         dense: [16]                 # LSTM is bidirectional
         bidirectional: True

    sequences:
       duration: 3.2                 # this experiments relies
       step: 0.8                     # on sliding windows of 3.2s
                                     # with a step of 0.8s
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
    This will create the following directory that contains a file called
    "tune.yml" describing the best hyper-parameters to use:

        <train_dir>/tune/<database.task.protocol>.<subset>

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequence "apply" mode.

"apply" mode
    Finally, one can apply speech activity detection using "apply" mode.
    This will create the following files that contains the hard and soft
    outputs of speech activity detection.

        <tune_dir>/apply/<database.task.protocol>.<subset>/{uri}.hard.json
                                                          /{uri}.soft.pkl
                                                          /eval.txt

    This means that file whose unique resource identifier is {uri} has been
    processed.

"""

import yaml
import pickle
import os.path
import functools
import numpy as np

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyannote.core
import pyannote.core.json

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.speech import SpeechActivityDetectionBatchGenerator

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Binarize

from pyannote.database import get_database
from pyannote.audio.optimizers import SSMORMS3

import skopt
import skopt.utils
import skopt.space
import skopt.plots
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics import f_measure


def train(protocol, experiment_dir, train_dir, subset='train'):

    # -- TRAINING --
    batch_size = 1024
    nb_epoch = 1000
    optimizer = SSMORMS3()

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features.yaafe',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    # -- ARCHITECTURE --
    architecture_name = config['architecture']['name']
    models = __import__('pyannote.audio.labeling.models',
                        fromlist=[architecture_name])
    Architecture = getattr(models, architecture_name)
    architecture = Architecture(
        **config['architecture'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']
    generator = SpeechActivityDetectionBatchGenerator(
        feature_extraction,
        duration=duration, step=step, batch_size=batch_size)

    # number of samples per epoch + round it to closest batch
    seconds_per_epoch = protocol.stats(subset)['annotated']
    samples_per_epoch = batch_size * \
        int(np.ceil((seconds_per_epoch / step) / batch_size))

    # input shape (n_frames, n_features)
    input_shape = generator.shape

    labeling = SequenceLabeling()
    labeling.fit(input_shape, architecture,
                 generator(getattr(protocol, subset)(), infinite=True),
                 samples_per_epoch, nb_epoch,
                 optimizer=optimizer, log_dir=train_dir)


def tune(protocol, train_dir, tune_dir, beta=1.0, subset='development'):

    np.random.seed(1337)
    os.makedirs(tune_dir)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'

    nb_epoch = 0
    while True:
        weights_h5 = WEIGHTS_H5.format(epoch=nb_epoch)
        if not os.path.isfile(weights_h5):
            break
        nb_epoch += 1

    config_dir = os.path.dirname(os.path.dirname(train_dir))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features.yaafe',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']

    predictions = {}

    def objective_function(parameters, beta=1.0):

        epoch, onset, offset = parameters

        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_labeling = SequenceLabeling.from_disk(
            architecture_yml, weights_h5)

        aggregation = SequenceLabelingAggregation(
            sequence_labeling, feature_extraction,
            duration=duration, step=step)

        if epoch not in predictions:
            predictions[epoch] = {}

        # no need to use collar during tuning
        precision = DetectionPrecision()
        recall = DetectionRecall()

        f, n = 0., 0
        for dev_file in getattr(protocol, subset)():

            uri = dev_file['uri']
            reference = dev_file['annotation']
            uem = dev_file['annotated']
            n += 1

            if uri in predictions[epoch]:
                prediction = predictions[epoch][uri]
            else:
                wav = dev_file['medium']['wav']
                prediction = aggregation.apply(wav)
                predictions[epoch][uri] = prediction

            binarizer = Binarize(onset=onset, offset=offset)
            hypothesis = binarizer.apply(prediction, dimension=1)

            p = precision(reference, hypothesis, uem=uem)
            r = recall(reference, hypothesis, uem=uem)
            f += f_measure(p, r, beta=beta)

        return 1 - (f / n)

    def callback(res):

        n_trials = len(res.func_vals)

        # save best parameters so far
        epoch, onset, offset = res.x
        params = {'epoch': int(epoch),
                  'onset': float(onset),
                  'offset': float(offset)}
        with open(tune_dir + '/tune.yml', 'w') as fp:
            yaml.dump(params, fp, default_flow_style=False)

        # plot convergence
        _ = skopt.plots.plot_convergence(res)
        plt.savefig(tune_dir + '/convergence.png', dpi=150)
        plt.close()

        if n_trials % 10 > 0:
            return

        # plot evaluations
        _ = skopt.plots.plot_evaluations(res)
        plt.savefig(tune_dir + '/evaluation.png', dpi=150)
        plt.close()

        try:
            # plot objective function
            _ = skopt.plots.plot_objective(res)
            plt.savefig(tune_dir + '/objective.png', dpi=150)
            plt.close()
        except Exception as e:
            pass

        # save results so far
        func = res['specs']['args']['func']
        callback = res['specs']['args']['callback']
        del res['specs']['args']['func']
        del res['specs']['args']['callback']
        skopt.utils.dump(res, tune_dir + '/tune.gz', store_objective=True)
        res['specs']['args']['func'] = func
        res['specs']['args']['callback'] = callback

    epoch = skopt.space.Integer(0, nb_epoch - 1)
    onset = skopt.space.Real(0., 1., prior='uniform')
    offset = skopt.space.Real(0., 1., prior='uniform')

    res = skopt.gp_minimize(
        functools.partial(objective_function, beta=beta),
        [epoch, onset, offset], callback=callback,
        n_calls=1000, n_random_starts=10,
        x0=[nb_epoch - 1, 0.5, 0.5],
        random_state=1337, verbose=True)

    return res


def test(protocol, tune_dir, apply_dir, subset='test', beta=1.0):

    os.makedirs(apply_dir)

    train_dir = os.path.dirname(os.path.dirname(tune_dir))
    config_dir = os.path.dirname(os.path.dirname(train_dir))

    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features.yaafe',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']

    # -- HYPER-PARAMETERS --

    tune_yml = tune_dir + '/tune.yml'
    with open(tune_yml, 'r') as fp:
        tune = yaml.load(fp)

    architecture_yml = train_dir + '/architecture.yml'
    WEIGHTS_H5 = train_dir + '/weights/{epoch:04d}.h5'
    weights_h5 = WEIGHTS_H5.format(epoch=tune['epoch'])

    sequence_labeling = SequenceLabeling.from_disk(
        architecture_yml, weights_h5)

    aggregation = SequenceLabelingAggregation(
        sequence_labeling, feature_extraction,
        duration=duration, step=step)

    binarizer = Binarize(onset=tune['onset'], offset=tune['offset'])

    HARD_JSON = apply_dir + '/{uri}.hard.json'
    SOFT_PKL = apply_dir + '/{uri}.soft.pkl'

    eval_txt = apply_dir + '/eval.txt'
    TEMPLATE = '{uri} {precision:.5f} {recall:.5f} {f_measure:.5f}\n'
    precision = DetectionPrecision()
    recall = DetectionRecall()
    fscore = []

    for test_file in getattr(protocol, subset)():

        uri = test_file['uri']
        wav = test_file['medium']['wav']
        soft = aggregation.apply(wav)
        hard = binarizer.apply(soft, dimension=1)

        with open(SOFT_PKL.format(uri=uri), 'w') as fp:
            pickle.dump(soft, fp)

        with open(HARD_JSON.format(uri=uri), 'w') as fp:
            pyannote.core.json.dump(hard, fp)

        try:
            reference = test_file['annotation']
            uem = test_file['annotated']
        except KeyError as e:
            continue

        p = precision(reference, hard, uem=uem)
        r = recall(reference, hard, uem=uem)
        f = f_measure(p, r, beta=beta)
        fscore.append(f)

        line = TEMPLATE.format(
            uri=uri, precision=p, recall=r, f_measure=f)
        with open(eval_txt, 'a') as fp:
            fp.write(line)

    p = abs(precision)
    r = abs(recall)
    f = np.mean(fscore)
    line = TEMPLATE.format(
        uri='ALL', precision=p, recall=r, f_measure=f)
    with open(eval_txt, 'a') as fp:
        fp.write(line)


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speech activity detection')

    medium_template = {}
    if '<wav_template>' in arguments:
        medium_template = {'wav': arguments['<wav_template>']}

    if '<database.task.protocol>' in arguments:
        protocol = arguments['<database.task.protocol>']
        database_name, task_name, protocol_name = protocol.split('.')
        database = get_database(database_name, medium_template=medium_template)
        protocol = database.get_protocol(task_name, protocol_name)

    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'train'
        train_dir = experiment_dir + '/train/' + arguments['<database.task.protocol>'] + '.' + subset
        train(protocol, experiment_dir, train_dir, subset=subset)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        beta = float(arguments.get('--recall'))
        tune_dir = train_dir + '/tune/' + arguments['<database.task.protocol>'] + '.' + subset
        res = tune(protocol, train_dir, tune_dir, beta=beta, subset=subset)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']
        if subset is None:
            subset = 'test'
        beta = float(arguments.get('--recall'))
        apply_dir = tune_dir + '/apply/' + arguments['<database.task.protocol>'] + '.' + subset
        res = test(protocol, tune_dir, apply_dir, beta=beta, subset=subset)
