#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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
# Ruiqing YIN

"""
Speaker change detection

Usage:
    pyannote-change-detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
    pyannote-change-detection evaluate [--database=<db.yml> --subset=<subset> --epoch=<epoch> --min_duration=<min_duration>] <train_dir> <database.task.protocol>
    pyannote-change-detection apply  [--database=<db.yml> --subset=<subset> --threshold=<threshold> --epoch=<epoch>  --min_duration=<min_duration>] <train_dir> <database.task.protocol>
    pyannote-change-detection -h | --help
    pyannote-change-detection --version

Options:
    <experiment_dir>                Set experiment root directory. This script expects
                                    a configuration file called "config.yml" to live
                                    in this directory. See "Configuration file"
                                    section below for more details.
    <database.task.protocol>        Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
    <wav_template>                  Set path to actual wave files. This path is
                                    expected to contain a {uri} placeholder that will
                                    be replaced automatically by the actual unique
                                    resource identifier (e.g. '/Etape/{uri}.wav').
    <train_dir>                     Set path to the directory containing pre-trained
                                    models (i.e. the output of "train" mode).
    --database=<db.yml>             Path to database configuration file.
                                    [default: ~/.pyannote/db.yml]
    --subset=<subset>               Set subset (train|developement|test).
                                    In "train" mode, default is "train".
                                    In "validation" mode, default is "development".
                                    In "tune" mode, default is "development".
                                    In "apply" mode, default is "test".
    --epoch=<epoch>                 The epoch in training process
    --threshold=<threshold>         Threshold for choosing change points
                                    [default: 0.1]
    --min_duration=<min_duration>   min duration between two adjacent peaks
                                    [default: 1.0]
    -h --help                       Show this screen.
    --version                       Show version.


Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
        The configuration of each experiment is described in a file called
        <experiment_dir>/config.yml, that describes the architecture of the neural
        network used for sequence labeling, the feature extraction process
        (e.g. MFCCs) and the sequence generator used for both training and applying.

        ................... <experiment_dir>/config.yml ...................
        feature_extraction:
             name: YaafeMFCC
             params:
                    e: False                   # this experiments relies
                    De: True                   # on 11 MFCC coefficients
                    DDe: True                  # with 1st and 2nd derivatives
                    D: True                    # without energy, but with
                    DD: True                   # energy derivatives
                    stack: 1
                    duration: 0.025
                    step: 0.010

        architecture:
             name: StackedLSTM
             params:                         # this experiments relies
                 n_classes: 1                # on one LSTM layer (16 outputs)
                 lstm: [16]                  # and one dense layer.
                 mlp: [16]                   # LSTM is bidirectional
                 bidirectional: 'concat'
                 final_activation: 'sigmoid'

        sequences:
             duration: 3.2                 # this experiments relies
             step: 0.8                     # on sliding windows of 3.2s
             balance: 0.05                 # with a step of 0.8s
             batch_size: 1024
        ...................................................................

"train" mode:
        First, one should train the raw sequence labeling neural network using
        "train" mode. This will create the following directory that contains
        the pre-trained neural network weights after each epoch:
                <experiment_dir>/train/<database.task.protocol>.<subset>

        This means that the network was trained on the <subset> subset of the
        <database.task.protocol> protocol. By default, <subset> is "train".
        This directory is called <train_dir> in the subsequent "evaluate" and "apply" mode.

"evaluate" mode:
        Then, one can evaluate the model with "evaluate" mode. This will create
        the following directory that contains coverages and purities based
        on different thresholds:
                <train_dir>/evaluate/<database.task.protocol>.<subset>
        This means that the model is evaluated on the <subset> subset of the
        <database.task.protocol> protocol. By default, <subset> is "development".

"apply" mode
        Finally, one can apply speaker change detection using "apply" mode.
        This will create the following files that contains the segmentation results
        with a requested threshold:
                <train_dir>/segments/<database.task.protocol>.<subset>/<threshold>/{uri}.0.seg
        This means that file whose unique resource identifier is {uri} has been
        processed.

"""

import yaml
import pickle
import os.path
import functools
import numpy as np

from docopt import docopt

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.change import ChangeDetectionBatchGenerator

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Peak

from pyannote.database import get_database
from pyannote.audio.optimizers import SSMORMS3

from pyannote.audio.callback import LoggingCallback

from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
from pyannote.metrics import f_measure

from pyannote.database.util import FileFinder
from pyannote.database.util import get_unique_identifier
from pyannote.database import get_protocol

from pyannote.audio.util import mkdir_p



def train(protocol, experiment_dir, train_dir, subset='train'):

    # -- TRAINING --
    nb_epoch = 1000
    optimizer = SSMORMS3()

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
            config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
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
    batch_size = config['sequences'].get('batch_size', 1024)
    duration = config['sequences']['duration']
    step = config['sequences']['step']
    balance = config['sequences']['balance']
    generator = ChangeDetectionBatchGenerator(
            feature_extraction, batch_size=batch_size,
            duration=duration, step=step, balance=balance)

    # number of steps per epoch
    seconds_per_epoch = protocol.stats(subset)['annotated']
    steps_per_epoch = int(np.ceil((seconds_per_epoch / step) / batch_size))

    # input shape (n_frames, n_features)
    input_shape = generator.shape

    labeling = SequenceLabeling()
    labeling.fit(input_shape, architecture,
                generator(getattr(protocol, subset)(), infinite=True),
                steps_per_epoch, nb_epoch, loss='binary_crossentropy',
                optimizer=optimizer, log_dir=train_dir)


def evaluate(protocol, train_dir, store_dir, subset='development',
    epoch=None, min_duration=1.0):

    mkdir_p(store_dir)

    # -- LOAD MODEL --
    nb_epoch = 0
    while True:
        weights_h5 = LoggingCallback.WEIGHTS_H5.format(log_dir=train_dir,
                                                       epoch=nb_epoch)
        if not os.path.isfile(weights_h5):
            break
        nb_epoch += 1
    config_dir = os.path.dirname(os.path.dirname(train_dir))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
        fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
            **config['feature_extraction'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']

    groundtruth = {}
    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        groundtruth[uri] = dev_file['annotation']

    # -- CHOOSE MODEL --
    if epoch > nb_epoch:
        raise ValueError('Epoch should be less than ' + str(nb_epoch))
    if epoch is None:
        epoch = nb_epoch - 1

    sequence_labeling = SequenceLabeling.from_disk(
            train_dir, epoch)

    aggregation = SequenceLabelingAggregation(
            sequence_labeling, feature_extraction,
            duration=duration, step=step)

    # -- PREDICTION --
    predictions = {}
    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        predictions[uri] = aggregation.apply(dev_file)

    alphas = np.linspace(0, 1, 20)

    purity = [SegmentationPurity(parallel=False) for alpha in alphas]
    coverage = [SegmentationCoverage(parallel=False) for alpha in alphas]

    # -- SAVE RESULTS --
    for i, alpha in enumerate(alphas):
        # initialize peak detection algorithm
        peak = Peak(alpha=alpha, min_duration=min_duration)
        for uri, reference in groundtruth.items():
            # apply peak detection
            hypothesis = peak.apply(predictions[uri])
            # compute purity and coverage
            purity[i](reference, hypothesis)
            coverage[i](reference, hypothesis)

    TEMPLATE = '{alpha:g} {purity:.3f}% {coverage:.3f}%'
    with open(store_dir + '/res.txt', 'a') as fp:
        for i, a in enumerate(alphas):
            p = 100 * abs(purity[i])
            c = 100 * abs(coverage[i])
            print(TEMPLATE.format(alpha=a, purity=p, coverage=c))
            fp.write(TEMPLATE.format(alpha=a, purity=p, coverage=c)+'\n')


def apply(protocol, train_dir, store_dir, threshold, subset='development',
    epoch=None, min_duration=1.0):

    # -- LOAD MODEL --
    nb_epoch = 0
    while True:
        weights_h5 = LoggingCallback.WEIGHTS_H5.format(log_dir=train_dir,
                                                       epoch=nb_epoch)
        if not os.path.isfile(weights_h5):
            break
        nb_epoch += 1
    config_dir = os.path.dirname(os.path.dirname(train_dir))
    config_yml = config_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
            config = yaml.load(fp)

    # -- FEATURE EXTRACTION --
    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features',
        fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
            **config['feature_extraction'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']

    def saveSeg(filepath,filename, segmentation):
        f = open(filepath,'w')
        for idx, val in enumerate(segmentation):
            line = filename + ' ' + str(idx) + ' 1 ' + str(int(val[0]*100))+' '+str(int(val[1]*100-val[0]*100))+'\n'
            f.write(line)
        f.close()

    filepath = store_dir+'/'+str(threshold) +'/'
    mkdir_p(filepath)

    # -- CHOOSE MODEL --
    if epoch > nb_epoch:
        raise ValueError('Epoch should be less than ' + str(nb_epoch))
    if epoch is None:
        epoch = nb_epoch - 1
    sequence_labeling = SequenceLabeling.from_disk(
            train_dir, epoch)
    aggregation = SequenceLabelingAggregation(
            sequence_labeling, feature_extraction,
            duration=duration, step=step)

    # -- PREDICTION --
    predictions = {}
    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        predictions[uri] = aggregation.apply(dev_file)

    # initialize peak detection algorithm
    peak = Peak(alpha=threshold, min_duration=min_duration)

    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        hypothesis = peak.apply(predictions[uri])
        filepath = store_dir+'/'+str(threshold) +'/'+uri+'.0.seg'
        saveSeg(filepath, uri, hypothesis)


def main():

    arguments = docopt(__doc__, version='Speaker change detection')
    db_yml = os.path.expanduser(arguments['--database'])
    preprocessors = {'audio': FileFinder(db_yml)}

    if '<database.task.protocol>' in arguments:
        protocol_name = arguments['<database.task.protocol>']
        protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'train'
        train_dir = experiment_dir + '/train/' + arguments['<database.task.protocol>'] + '.' + subset
        train(protocol, experiment_dir, train_dir, subset=subset)


    if arguments['evaluate']:
        train_dir = arguments['<train_dir>']
        epoch = arguments['--epoch']
        min_duration= arguments['--min_duration']
        if subset is None:
            subset = 'development'
        if epoch is not None:
            epoch = int(epoch)

        min_duration = float(min_duration)
        store_dir = train_dir + '/evaluate/' + arguments['<database.task.protocol>'] + '.' + subset
        res = evaluate(protocol, train_dir, store_dir, subset=subset,
            epoch=epoch, min_duration=min_duration)


    if arguments['apply']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        threshold = arguments['--threshold']
        threshold = float(threshold)
        epoch = arguments['--epoch']
        if epoch is not None:
            epoch = int(epoch)
        min_duration= arguments['--min_duration']
        min_duration = float(min_duration)
        store_dir = train_dir + '/segments/' + arguments['<database.task.protocol>'] + '.' + subset
        res = apply(protocol, train_dir, store_dir, threshold, subset=subset,
            epoch=epoch, min_duration=min_duration)
