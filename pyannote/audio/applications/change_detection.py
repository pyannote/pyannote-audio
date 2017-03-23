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
# Ruiqing YIN

"""
change detection

Usage:
    pyannote-change-detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
    pyannote-change-detection evaluate [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
    pyannote-change-detection apply  [--database=<db.yml> --subset=<subset> --threshold=<threshold>] <train_dir> <database.task.protocol>
    pyannote-change-detection -h | --help
    pyannote-change-detection --version

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
    --database=<db.yml>        Path to database configuration file.
                               [default: ~/.pyannote/db.yml]
    --subset=<subset>          Set subset (train|developement|test).
                               In "train" mode, default is "train".
                               In "validation" mode, default is "development".
                               In "tune" mode, default is "development".
                               In "apply" mode, default is "test".
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
                    stack: 3
 
        architecture:
             name: StackedLSTM
             params:                         # this experiments relies
                 n_classes: 1                # on one LSTM layer (16 outputs)
                 lstm: [16]                  # and one dense layer.
                 mlp: [16]                   # LSTM is bidirectional
                 bidirectional: 'concat'
                 final_activation: 'sigmoid'

        type:
                name: type2
                params:
                    num_mfcc: 3

        sequences:
             duration: 3.2                 # this experiments relies
             step: 0.8                     # on sliding windows of 3.2s
             balance: 0.05                 # with a step of 0.8s
        ...................................................................

"train" mode:
        First, one should train the raw sequence labeling neural network using
        "train" mode. This will create the following directory that contains
        the pre-trained neural network weights after each epoch:
                <experiment_dir>/train/<database.task.protocol>.<subset>

        This means that the network was trained on the <subset> subset of the
        <database.task.protocol> protocol. By default, <subset> is "train".
        This directory is called <train_dir> in the subsequent "tune" mode.

"evaluate" mode:
        Then, one can evaluate the model with "evaluate" mode. This will create 
        the following directory that contains coverages and purities based 
        on different thresholds:
                <train_dir>/evaluate/<database.task.protocol>.<subset>
        This means that the model is evaluated on the <subset> subset of the
        <database.task.protocol> protocol. By default, <subset> is "development".

"apply" mode
        Finally, one can apply speaker change detection using "segment" mode.
        This will create the following files that contains the segmentation results:
                <tune_dir>/segments/<database.task.protocol>.<subset>/threshold/{uri}.0.seg
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
from pyannote.audio.generators.change import ChangeDetectionBatchGenerator

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Peak

from pyannote.database import get_database
from pyannote.audio.optimizers import SSMORMS3

import skopt
import skopt.utils
import skopt.space
import skopt.plots
from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
from pyannote.metrics import f_measure

from pyannote.database.util import FileFinder
from pyannote.database.util import get_unique_identifier


def train(protocol, experiment_dir, train_dir, subset='train'):

        # -- TRAINING --
        batch_size = 1024
        nb_epoch = 2
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
        duration = config['sequences']['duration']
        step = config['sequences']['step']
        balance = config['sequences']['balance']
        generator = ChangeDetectionBatchGenerator(
                feature_extraction,
                duration=duration, step=step, batch_size=batch_size,balance=balance)

        # number of samples per epoch + round it to closest batch
        seconds_per_epoch = protocol.stats(subset)['annotated']
        samples_per_epoch = batch_size * \
                int(np.ceil((seconds_per_epoch / step) / batch_size))

        # input shape (n_frames, n_features)
        input_shape = generator.shape
 
        labeling = SequenceLabeling()
        labeling.fit(input_shape, architecture,
                                 generator(getattr(protocol, subset)(), infinite=True),
                                 samples_per_epoch, nb_epoch, loss='binary_crossentropy',
                                 optimizer=optimizer, log_dir=train_dir)



def evaluate(protocol, train_dir, store_dir, subset='development'):
    os.makedirs(store_dir)

    # -- LOAD MODEL --
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

    def objective_function(epoch):
        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_labeling = SequenceLabeling.from_disk(
                architecture_yml, weights_h5)


        aggregation = SequenceLabelingAggregation(
                sequence_labeling, feature_extraction,
                duration=duration, step=step)


        predictions = {}

        for dev_file in getattr(protocol, subset)():
            uri = dev_file['uri']
            #wav = dev_file['wav']
            predictions[uri] = aggregation.apply(dev_file)

        alphas = np.linspace(0, 1, 20)

        purity = [SegmentationPurity(parallel=False) for alpha in alphas]
        coverage = [SegmentationCoverage(parallel=False) for alpha in alphas]

        for i, alpha in enumerate(alphas):
            # initialize peak detection algorithm
            peak = Peak(alpha=alpha, min_duration=1.0)
            for uri, reference in groundtruth.items():
                # apply peak detection
                hypothesis = peak.apply(predictions[uri])
                # compute purity and coverage
                purity[i](reference, hypothesis)
                coverage[i](reference, hypothesis)

        TEMPLATE = '{alpha:.2f} {purity:.1f}% {coverage:.1f}%'
        res = []
        for i, a in enumerate(alphas):
            p = 100 * abs(purity[i])
            c = 100 * abs(coverage[i])
            print(TEMPLATE.format(alpha=a, purity=p, coverage=c))
            res.append((a,p,c))
        return res


    res = objective_function(nb_epoch-1)

    with open(store_dir + '/res.yml', 'w') as fp:
        yaml.dump(res, fp, default_flow_style=False)


def apply(protocol, train_dir, store_dir, threshold, subset='development'):
    #os.makedirs(store_dir)

    # -- LOAD MODEL --
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
    features = __import__('pyannote.audio.features',
                                                fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
            **config['feature_extraction'].get('params', {}))

    # -- SEQUENCE GENERATOR --
    duration = config['sequences']['duration']
    step = config['sequences']['step']

    def saveSeg(filepath,filename,chn,segmentation):
        f = open(filepath,'w')
        for idx, val in enumerate(segmentation):
                line = filename+' '+str(idx)+' '+str(chn)+' '+str(int(val[0]*100))+' '+str(int(val[1]*100-val[0]*100))+'\n'
                f.write(line)
        f.close()

    def get_aggregation(epoch):
        weights_h5 = WEIGHTS_H5.format(epoch=epoch)
        sequence_labeling = SequenceLabeling.from_disk(
                architecture_yml, weights_h5)


        aggregation = SequenceLabelingAggregation(
                sequence_labeling, feature_extraction,
                duration=duration, step=step)

        return aggregation


    filepath = store_dir+'/'+str(threshold) +'/'
    os.makedirs(filepath)

    aggregation = get_aggregation(nb_epoch-1)

    predictions = {}
    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        predictions[uri] = aggregation.apply(dev_file)  

    # initialize peak detection algorithm
    peak = Peak(alpha=threshold, min_duration=2.5)

    for dev_file in getattr(protocol, subset)():
        uri = dev_file['uri']
        prediction = predictions[uri]
        hypothesis = peak.apply(prediction)
        filepath = store_dir+'/'+str(threshold) +'/'+uri+'.0.seg'
        chn = 1
        saveSeg(filepath,uri,chn,hypothesis)


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speaker change detection')
    db_yml = os.path.expanduser(arguments['--database'])
    preprocessors = {'wav': FileFinder(db_yml)}

    if '<database.task.protocol>' in arguments:
        protocol = arguments['<database.task.protocol>']
        database_name, task_name, protocol_name = protocol.split('.')
        database = get_database(database_name, preprocessors=preprocessors)
        protocol = database.get_protocol(task_name, protocol_name, progress=True)

    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']
        if subset is None:
            subset = 'train'
        train_dir = experiment_dir + '/train/' + arguments['<database.task.protocol>'] + '.' + subset
        train(protocol, experiment_dir, train_dir, subset=subset)


    if arguments['evaluate']:
        train_dir = arguments['<train_dir>']
        if subset is None:
                subset = 'development'
        store_dir = train_dir + '/evaluate/' + arguments['<database.task.protocol>'] + '.' + subset
        res = evaluate(protocol, train_dir, store_dir, subset=subset)


    if arguments['apply']:
        train_dir = arguments['<train_dir>']
        if subset is None:
            subset = 'development'
        threshold = arguments['--threshold']
        if threshold is None:
            threshold = 0.7
        else:
            threshold = float(threshold)
        store_dir = train_dir + '/segments/' + arguments['<database.task.protocol>'] + '.' + subset
        res = apply(protocol, train_dir, store_dir, threshold, subset=subset)

