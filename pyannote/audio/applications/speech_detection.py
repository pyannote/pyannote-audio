#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
  pyannote-speech-detection train [options] <experiment_dir> <database.task.protocol>
  pyannote-speech-detection validate [options] [--every=<epoch> --chronological] <train_dir> <database.task.protocol>
  pyannote-speech-detection apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-speech-detection -h | --help
  pyannote-speech-detection --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default subset is "train".
                             In "validate" mode, defaults to "development".
                             In "apply" mode, defaults to "test".
  --gpu                      Run on GPUs. Defaults to using CPUs.
  --batch=<size>             Set batch size. Has no effect in "train" mode.
                             [default: 32]
  --from=<epoch>             Start {train|validat}ing at epoch <epoch>. Has no
                             effect in "apply" mode. [default: 0]
  --to=<epochs>              End {train|validat}ing at epoch <epoch>.
                             Defaults to keep going forever.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  --chronological            Force validation in chronological order.
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).

"apply" mode:
  <model.pt>                 Path to the pretrained model.
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: Precomputed
       params:
          root_dir: /path/to/mfcc

    architecture:
       name: StackedRNN
       params:
         rnn: LSTM
         recurrent: [16]
         bidirectional: True
         linear: [16]

    task:
       name: SpeechActivityDetection
       params:
          duration: 3.2
          batch_size: 32
          parallel: 2
    ...................................................................

"train" mode:
    First, one should train the raw sequence labeling neural network using
    "train" mode. This will create the following directory that contains
    the pre-trained neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

"validate" mode:
    In parallel to training, one should validate the performance of the model
    epoch after epoch, using "validate" mode. This will create a bunch of files
    in the following directory:

        <train_dir>/validate/<database.task.protocol>

    This means that the network was validated on the <database.task.protocol>
    protocol. By default, validation is done on the "development" subset:
    "developement.DetectionErrorRate.{txt|png|eps}" files are created and
    updated continuously, epoch after epoch. This directory is called
    <validate_dir> in the subsequent "tune" mode.

    In practice, for each epoch, "validate" mode will report the detection
    error rate obtained by applying (possibly not optimal) onset/offset
    thresholds of 0.5.

"apply" mode
    Finally, one can apply speech activity detection using "apply" mode.
    Created files can then be used in the following way:

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> from pyannote.audio.signal import Binarize
    >>> binarizer = Binarize()

    >>> raw_scores = precomputed(first_test_file)
    >>> speech_regions = binarizer.apply(raw_scores, dimension=1)

"""

import numpy as np
from docopt import docopt
from .base import Application
from os.path import expanduser
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.signal import Binarize
from pyannote.database import get_annotated
from pyannote.audio.features.utils import Precomputed
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.audio.labeling.extraction import SequenceLabeling


class SpeechActivityDetection(Application):

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeechActivityDetection, self).__init__(
            experiment_dir, db_yml=db_yml)

        # task
        task_name = self.config_['task']['name']
        tasks = __import__('pyannote.audio.labeling.tasks',
                           fromlist=[task_name])
        Task = getattr(tasks, task_name)
        self.task_ = Task(
            **self.config_['task'].get('params', {}))

        n_features = int(self.feature_extraction_.dimension())
        n_classes = self.task_.n_classes

        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.labeling.models',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        self.model_ = Architecture(
            n_features, n_classes,
            **self.config_['architecture'].get('params', {}))

    def train(self, protocol_name, subset='train', restart=None, epochs=1000):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        self.task_.fit(self.model_, self.feature_extraction_, protocol,
                       train_dir, subset=subset, epochs=epochs,
                       restart=restart, gpu=self.gpu)

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):


        # load model for current epoch
        model = self.load_model(epoch)
        if self.gpu:
            model = model.cuda()
        model.eval()

        der = DetectionErrorRate()

        binarizer = Binarize(onset=0.5, offset=0.5,
                             log_scale=False)

        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        duration = self.task_.duration
        step = .25 * duration
        sequence_labeling = SequenceLabeling(
            model, self.feature_extraction_, duration,
            step=.25 * duration, batch_size=self.batch_size,
            source='audio', gpu=self.gpu)

        sequence_labeling.cache_preprocessed_ = False

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        file_generator = getattr(protocol, subset)()
        for current_file in file_generator:

            predictions = sequence_labeling.apply(current_file)

            if model.logsoftmax:
                predictions = SlidingWindowFeature(
                    1. - np.exp(predictions.data[:, 0]),
                    predictions.sliding_window)
            else:
                predictions = SlidingWindowFeature(
                    1. - predictions.data[:, 0],
                    predictions.sliding_window)

            hypothesis = binarizer.apply(
                predictions, dimension=0).to_annotation()

            reference = current_file['annotation']
            uem = get_annotated(current_file)
            _ = der(reference, hypothesis, uem=uem)

        return {'DetectionErrorRate': {'minimize': True, 'value': abs(der)}}

    def apply(self, protocol_name, output_dir, step=None):

        model = self.model_

        if self.gpu:
            model = model.cuda()
        model.eval()

        duration = self.task_.duration
        if step is None:
            step = 0.25 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model, self.feature_extraction_, duration,
            step=.25 * duration, batch_size=self.batch_size,
            source='audio', gpu=self.gpu)

        sliding_window = sequence_labeling.sliding_window
        n_classes = self.task_.n_classes

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=n_classes)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for current_file in FileFinder.protocol_file_iter(
            protocol, extra_keys=['audio']):

            fX = sequence_labeling.apply(current_file)
            precomputed.dump(current_file, fX)


def main():

    arguments = docopt(__doc__, version='Speech activity detection')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        # start training at this epoch (defaults to 0)
        restart = int(arguments['--from'])

        # stop training at this epoch (defaults to never stop)
        epochs = arguments['--to']
        if epochs is None:
            epochs = np.inf
        else:
            epochs = int(epochs)

        application = SpeechActivityDetection(experiment_dir, db_yml=db_yml)
        application.gpu = gpu
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)

    if arguments['validate']:
        train_dir = arguments['<train_dir>']

        if subset is None:
            subset = 'development'

        # start validating at this epoch (defaults to 0)
        start = int(arguments['--from'])

        # stop validating at this epoch (defaults to np.inf)
        end = arguments['--to']
        if end is None:
            end = np.inf
        else:
            end = int(end)

        # validate every that many epochs (defaults to 1)
        every = int(arguments['--every'])

        # validate epochs in chronological order
        in_order = arguments['--chronological']

        # batch size
        batch_size = int(arguments['--batch'])

        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.gpu = gpu
        application.batch_size = batch_size
        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order)

    if arguments['apply']:

        model_pt = arguments['<model.pt>']
        output_dir = arguments['<output_dir>']
        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = SpeechActivityDetection.from_model_pt(
            model_pt, db_yml=db_yml)
        application.gpu = gpu
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step)
