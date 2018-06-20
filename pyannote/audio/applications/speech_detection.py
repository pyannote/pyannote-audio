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
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Not used in
                             "apply" mode.
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
    # train the network for speech activity detection
    # see pyannote.audio.labeling.tasks for more details
    task:
       name: SpeechActivityDetection
       params:
          duration: 3.2     # sub-sequence duration
          per_epoch: 36000  # 10 hours of audio per epoch
          batch_size: 32    # number of sub-sequences per batch
          parallel: 4       # number of background generators

    # use precomputed features (see feature extraction tutorial)
    feature_extraction:
       name: Precomputed
       params:
          root_dir: tutorials/feature-extraction

    # use the StackedRNN architecture.
    # see pyannote.audio.labeling.models for more details
    architecture:
       name: StackedRNN
       params:
         rnn: LSTM
         recurrent: [16, 16]
         bidirectional: True

    # use cyclic learning rate scheduler
    scheduler:
       name: CyclicScheduler
       params:
           learning_rate: auto
    ...................................................................

"train" mode:
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

    A bunch of values (loss, learning rate, ...) are sent to and can be
    visualized with tensorboard with the following command:

        $ tensorboard --logdir=<experiment_dir>

"validate" mode:
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results:

        <train_dir>/validate/<database.task.protocol>.<subset>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).

    In practice, for each epoch, "validate" mode will look for the detection
    threshold that minimizes the detection error rate. Both values (best
    threshold and corresponding error rate) are sent to tensorboard.

"apply" mode:
    Use the "apply" mode to extract speech activity detection raw scores.
    Resulting files can then be used in the following way:

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

import torch
import numpy as np
import scipy.optimize
from pathlib import Path
from docopt import docopt
from .base import Application
from os.path import expanduser
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.signal import Binarize
from pyannote.database import get_annotated
from pyannote.core import SlidingWindowFeature
from pyannote.database import get_unique_identifier
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

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        duration = self.task_.duration
        step = .25 * duration
        sequence_labeling = SequenceLabeling(
            model, self.feature_extraction_, duration=duration,
            step=.25 * duration, batch_size=self.batch_size,
            source='audio', device=self.device)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        metric = DetectionErrorRate()

        predictions = {}

        file_generator = getattr(protocol, subset)()
        for current_file in file_generator:
            uri = get_unique_identifier(current_file)
            scores = sequence_labeling.apply(current_file)

            if model.logsoftmax:
                scores = SlidingWindowFeature(
                    1. - np.exp(scores.data[:, 0]),
                    scores.sliding_window)
            else:
                scores = SlidingWindowFeature(
                    1. - scores.data[:, 0],
                    scores.sliding_window)

            predictions[uri] = scores

        def fun(threshold):

            binarizer = Binarize(onset=threshold,
                                 offset=threshold,
                                 log_scale=False)

            protocol = get_protocol(protocol_name, progress=False,
                                    preprocessors=self.preprocessors_)

            metric = DetectionErrorRate()

            file_generator = getattr(protocol, subset)()
            for current_file in file_generator:

                uri = get_unique_identifier(current_file)
                hypothesis = binarizer.apply(
                    predictions[uri], dimension=0).to_annotation()
                reference = current_file['annotation']
                uem = get_annotated(current_file)
                _ = metric(reference, hypothesis, uem=uem)

            return abs(metric)

        res = scipy.optimize.minimize_scalar(
            fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

        return {
            'speech_activity_detection/error': {'minimize': True,
                                                'value': res.fun},
            'speech_activity_detection/threshold': {'minimize': 'NA',
                                                    'value': res.x}}

    def apply(self, protocol_name, output_dir, step=None):

        model = self.model_.to(self.device)
        model.eval()

        duration = self.task_.duration
        if step is None:
            step = 0.25 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_labeling = SequenceLabeling(
            model, self.feature_extraction_, duration=duration,
            step=.25 * duration, batch_size=self.batch_size,
            source='audio', device=self.device)

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

    db_yml = Path(arguments['--database'])
    db_yml = db_yml.expanduser().resolve(strict=True)

    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')

    if arguments['train']:
        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

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
        application.device = device
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)

    if arguments['validate']:

        train_dir = Path(arguments['<train_dir>'])
        train_dir = train_dir.expanduser().resolve(strict=True)

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
        application.device = device
        application.batch_size = batch_size
        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order)

    if arguments['apply']:

        model_pt = Path(arguments['<model_pt>'])
        model_pt = model_pt.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = SpeechActivityDetection.from_model_pt(
            model_pt, db_yml=db_yml)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step)
