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

"train" mode expects a file <config_dir>/config.yml and will create the following
directory structure:
   <train_dir>/database.task.protocol/architecture.yml
                                       weights/{epoch:04d}.h5
where <train_dir> = <config_dir>/database.task.protocol

"tune" m

<train_dir>

# apply

Usage:
  speech_activity_detection train <experiment_dir> <database.task.protocol> <wav_template>
  speech_activity_detection tune <train_dir> <database.task.protocol> <wav_template>
  speech_activity_detection -h | --help
  speech_activity_detection --version

Options:
  <experiment_dir>               Directory where config.yml is stored.
  <database.task.protocol>       Evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  <wav_template>                 Template to actual media files (e.g. '/Users/bredin/Corpora/etape/{uri}.wav')
  <train_dir>                    Directory where train mode output its files
  <output_dir>                   Path where to save results.
  -h --help                      Show this screen.
  --version                      Show version.

"""

import yaml
import os.path
import numpy as np
from docopt import docopt

import pyannote.core

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(protocol, experiment_dir, train_dir):

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
    normalize = config['sequences']['normalize']
    generator = SpeechActivityDetectionBatchGenerator(
        feature_extraction, normalize=normalize,
        duration=duration, step=step, batch_size=batch_size)

    # number of samples per epoch + round it to closest batch
    seconds_per_epoch = protocol.stats('train')['annotated']
    samples_per_epoch = batch_size * \
        int(np.ceil((seconds_per_epoch / step) / batch_size))

    # input shape (n_frames, n_features)
    input_shape = generator.get_shape()

    labeling = SequenceLabeling()
    labeling.fit(input_shape, architecture,
                 generator(protocol.train(), infinite=True),
                 samples_per_epoch, nb_epoch,
                 optimizer=optimizer, log_dir=train_dir)


def tune(protocol, train_dir, tune_dir):

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
    normalize = config['sequences']['normalize']

    predictions = {}

    def objective_function(parameters, beta=1.0):

        epoch, onset, offset = parameters

        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_labeling = SequenceLabeling.from_disk(
            architecture_yml, weights_h5)

        aggregation = SequenceLabelingAggregation(
            sequence_labeling, feature_extraction, normalize=normalize,
            duration=duration, step=step)

        if epoch not in predictions:
            predictions[epoch] = {}

        # no need to use collar during tuning
        precision = DetectionPrecision()
        recall = DetectionRecall()

        f, n = 0., 0
        for dev_file in protocol.development():

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

        if n_trials % 10 == 0:
            
            # plot evaluations
            _ = skopt.plots.plot_evaluations(res)
            plt.savefig(tune_dir + '/history.png', dpi=150)
            plt.close()

            # plot objective function
            #_ = skopt.plots.plot_objective(res)
            #plt.savefig(tune_dir + '/objective.png', dpi=150)
            #plt.close()

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
    dimensions = [epoch, onset, offset]
    res = skopt.gp_minimize(objective_function, dimensions,
                      n_calls=1000, n_random_starts=10,
                      x0=[nb_epoch - 1, 0.5, 0.5],
                      random_state=1337, verbose=True,
                      callback=callback)

    return res

# def test(dataset, medium_template, config_yml, weights_h5, output_dir):
#
#     # load configuration file
#     with open(config_yml, 'r') as fp:
#         config = yaml.load(fp)
#
#     # this is where model architecture was saved
#     architecture_yml = os.path.dirname(os.path.dirname(weights_h5)) + '/architecture.yml'
#
#     # -- DATASET --
#     db, task, protocol, subset = dataset.split('.')
#     database = get_database(db, medium_template=medium_template)
#     protocol = database.get_protocol(task, protocol)
#
#     if not hasattr(protocol, subset):
#         raise NotImplementedError('')
#
#     file_generator = getattr(protocol, subset)()
#
#     # -- FEATURE EXTRACTION --
#     # input sequence duration
#     duration = config['feature_extraction']['duration']
#     # MFCCs
#     feature_extraction = YaafeMFCC(**config['feature_extraction']['mfcc'])
#     # normalization
#     normalize = config['feature_extraction']['normalize']
#
#     # -- TESTING --
#     # overlap ratio between each window
#     overlap = config['testing']['overlap']
#     step = duration * (1. - overlap)
#
#     # prediction smoothing
#     onset = config['testing']['binarize']['onset']
#     offset = config['testing']['binarize']['offset']
#     binarizer = Binarize(onset=0.5, offset=0.5)
#
#     sequence_labeling = SequenceLabeling.from_disk(
#         architecture_yml, weights_h5)
#
#     aggregation = SequenceLabelingAggregation(
#         sequence_labeling, feature_extraction, normalize=normalize,
#         duration=duration, step=step)
#
#     collar = 0.500
#     error_rate = DetectionErrorRate(collar=collar)
#     accuracy = DetectionAccuracy(collar=collar)
#     precision = DetectionPrecision(collar=collar)
#     recall = DetectionRecall(collar=collar)
#
#     LINE = '{uri} {e:.3f} {a:.3f} {p:.3f} {r:.3f} {f:.3f}\n'
#
#     PATH = '{output_dir}/eval.{dataset}.{subset}.txt'
#     path = PATH.format(output_dir=output_dir, dataset=dataset, subset=subset)
#
#     with open(path, 'w') as fp:
#
#         header = '# uri error accuracy precision recall f_measure\n'
#         fp.write(header)
#         fp.flush()
#
#         for current_file in file_generator:
#
#             uri = current_file['uri']
#             wav = current_file['medium']['wav']
#             annotated = current_file['annotated']
#             annotation = current_file['annotation']
#
#             predictions = aggregation.apply(wav)
#             hypothesis = binarizer.apply(predictions, dimension=1)
#
#             e = error_rate(annotation, hypothesis, uem=annotated)
#             a = accuracy(annotation, hypothesis, uem=annotated)
#             p = precision(annotation, hypothesis, uem=annotated)
#             r = recall(annotation, hypothesis, uem=annotated)
#             f = f_measure(p, r)
#
#             line = LINE.format(uri=uri, e=e, a=a, p=p, r=r, f=f)
#             fp.write(line)
#             fp.flush()
#
#             PATH = '{output_dir}/{uri}.json'
#             path = PATH.format(output_dir=output_dir, uri=uri)
#             dump_to(hypothesis, path)
#
#         # average on whole corpus
#         uri = '{dataset}.{subset}'.format(dataset=dataset, subset=subset)
#         e = abs(error_rate)
#         a = abs(accuracy)
#         p = abs(precision)
#         r = abs(recall)
#         f = f_measure(p, r)
#         line = LINE.format(uri=uri, e=e, a=a, p=p, r=r, f=f)
#         fp.write(line)
#         fp.flush()


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

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        train_dir = experiment_dir + '/train/' + arguments['<database.task.protocol>']
        train(protocol, experiment_dir, train_dir)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']
        tune_dir = train_dir + '/tune/' + arguments['<database.task.protocol>']
        res = tune(protocol, train_dir, tune_dir)
