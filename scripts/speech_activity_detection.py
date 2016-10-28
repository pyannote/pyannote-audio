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
  speech_activity_detection train <config.yml> <database.task.protocol> <wav_template> 
  speech_activity_detection apply <config.yml> <weights.h5> <database.task.protocol> <wav_template> <output_dir>
  speech_activity_detection -h | --help
  speech_activity_detection --version

Options:
  <config.yml>              Use this configuration file.
  <database.task.protocol>  Use this dataset (e.g. "Etape.SpeakerDiarization.TV" for training)
  <wav_template>            Path template to actual media files (e.g. '/Users/bredin/Corpora/etape/{uri}.wav')
  <weights.h5>              Path to pre-trained model weights. File
                            'architecture.yml' must live in the same directory.
  <output_dir>              Path where to save results.
  -h --help                 Show this screen.
  --version                 Show version.

"""

import yaml
import os.path
import numpy as np
from docopt import docopt

import pyannote.core
from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.speech import SpeechActivityDetectionBatchGenerator
from pyannote.database import get_database
from pyannote.audio.optimizers import SSMORMS3

def train(protocol, medium_template, config_yml):

    # -- TRAINING --
    batch_size = 1024
    nb_epoch = 1000
    optimizer = SSMORMS3()

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # deduce workdir from path of configuration file
    workdir = os.path.dirname(config_yml)
    # this is where model weights are saved after each epoch
    LOG_DIR = workdir + '/' + protocol

    # -- PROTOCOL --
    database_name, task_name, protocol_name = protocol.split('.')
    database = get_database(database_name, medium_template=medium_template)
    protocol = database.get_protocol(task_name, protocol_name)

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

    # # log loss and accuracy during training and
    # # keep track of best models for both metrics
    # log = [('train', 'loss'), ('train', 'accuracy')]
    # callback = LoggingCallback(log_dir=log_dir, log=log)

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
                 optimizer=optimizer, log_dir=LOG_DIR)

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

    if arguments['train']:

        # arguments
        protocol = arguments['<database.task.protocol>']
        medium_template = {'wav': arguments['<wav_template>']}
        config_yml = arguments['<config.yml>']

        # train the model
        train(protocol, medium_template, config_yml)

    # if arguments['apply']:
    #
    #     # arguments
    #     config_yml = arguments['<config.yml>']
    #     weights_h5 = arguments['<weights.h5>']
    #     protocol = arguments['<database.task.protocol>']
    #     medium_template = {'wav': arguments['<wav_template>']}
    #     output_dir = arguments['<output_dir>']
    #
    #     test(protocol, medium_template, config_yml, weights_h5, output_dir)
