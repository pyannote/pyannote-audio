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
# Hervé BREDIN - http://herve.niderb.fr

"""
Speech activity detection

Usage:
  speech_activity_detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  speech_activity_detection validate [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
  speech_activity_detection tune  [--database=<db.yml> --subset=<subset>] <train_dir> <database.task.protocol>
  speech_activity_detection apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  speech_activity_detection -h | --help
  speech_activity_detection --version

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
                             Use higher values if you want to improve recall.
  -h --help                  Show this screen.
  --version                  Show version.


Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

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

import os.path
from docopt import docopt
import pyannote.core
from pyannote.audio.applications import SpeechActivityDetection

if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speech activity detection')

    db_yml = os.path.expanduser(arguments['--database'])
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
