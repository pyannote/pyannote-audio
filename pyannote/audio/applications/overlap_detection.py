#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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
Overlap detection

Usage:
  pyannote-overlap-detection train [options] <experiment_dir> <database.task.protocol>
  pyannote-overlap-detection validate [options] [--every=<epoch> --chronological --precision=<precision>] <train_dir> <database.task.protocol>
  pyannote-overlap-detection apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-overlap-detection -h | --help
  pyannote-overlap-detection --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<db.yml>        Path to database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to all subsets in
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
  --precision=<precision>    Target detection precision [default: 0.8].

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
    # train the network for overlap detection
    # see pyannote.audio.labeling.tasks for more details
    task:
       name: OverlapDetection
       params:
          duration: 3.2     # sub-sequence duration
          per_epoch: 1      # 1 day of audio per epoch
          collar: 0.200     # upsampling collar
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
         recurrent: [32, 20]
         bidirectional: True
         linear: [40, 10]

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

    In practice, for each epoch, "validate" mode will look for the peak
    detection threshold that maximizes speech turn coverage, under the
    constraint that purity must be greater than the value provided by the
    "--purity" option. Both values (best threshold and corresponding coverage)
    are sent to tensorboard.

"apply" mode
    Use the "apply" mode to extract speaker change detection raw scores.
    Resulting files can then be used in the following way:

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> from pyannote.audio.signal import Binarize
    >>> binarizer = Binarize()

    >>> raw_scores = precomputed(first_test_file)
    >>> overlap_regions = binarizer.apply(raw_scores, dimension=1)
"""


from functools import partial
from pathlib import Path

import numpy as np
import torch
from docopt import docopt

from pyannote.core import Timeline

from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier

from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision

from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.signal import Binarize
from .speech_detection import SpeechActivityDetection

from pyannote.audio.pipeline.overlap_detection \
    import OverlapDetection as OverlapDetectionPipeline


def validate_helper_func(current_file, pipeline=None, precision=None, recall=None):
    reference = current_file['annotation']
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file)
    p = precision(reference, hypothesis, uem=uem)
    r = recall(reference, hypothesis, uem=uem)
    return p, r

class OverlapDetection(SpeechActivityDetection):

    def validate_init(self, protocol_name, subset='development'):
        validation_data = super().validate_init(protocol_name, subset=subset)
        for current_file in validation_data:

            uri = current_file['uri']

            # build overlap reference
            overlap = Timeline(uri=uri)
            turns = current_file['annotation']
            for track1, track2 in turns.co_iter(turns):
                if track1 == track2:
                    continue
                overlap.add(track1[0] & track2[0])
            current_file['annotation'] = overlap.support().to_annotation()

        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) overlap scores
        duration = self.task_.duration
        step = .25 * duration
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)
        for current_file in validation_data:
            uri = get_unique_identifier(current_file)
            current_file['ovl_scores'] = sequence_labeling(current_file)

        # pipeline
        pipeline = OverlapDetectionPipeline(precision=self.precision)

        # dichotomic search to find threshold that maximizes recall
        # while having at least `target_precision`

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        for _ in range(10):

            current_alpha = .5 * (lower_alpha + upper_alpha)
            pipeline.instantiate({'onset': current_alpha,
                                  'offset': current_alpha,
                                  'min_duration_on': 0.,
                                  'min_duration_off': 0.,
                                  'pad_onset': 0.,
                                  'pad_offset': 0.})

            precision = DetectionPrecision(parallel=True)
            recall = DetectionRecall(parallel=True)

            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               precision=precision,
                               recall=recall)
            _ = self.pool_.map(validate, validation_data)

            precision = abs(precision)
            recall = abs(recall)

            if precision < self.precision:
                # precision is not high enough: try higher thresholds
                lower_alpha = current_alpha

            else:
                upper_alpha = current_alpha
                if recall > best_recall:
                    best_recall = recall
                    best_alpha = current_alpha

        return {'metric': f'recall@{self.precision:.2f}precision',
                'minimize': False,
                'value': best_recall,
                'pipeline': pipeline.instantiate({'onset': best_alpha,
                                                  'offset': best_alpha,
                                                  'min_duration_on': 0.,
                                                  'min_duration_off': 0.,
                                                  'pad_onset': 0.,
                                                  'pad_offset': 0.})}

def main():

    arguments = docopt(__doc__, version='Overlap detection')

    db_yml = arguments['--database']
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

        application = OverlapDetection(experiment_dir, db_yml=db_yml,
                                       training=True)
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

        precision = float(arguments['--precision'])

        application = OverlapDetection.from_train_dir(
            train_dir, db_yml=db_yml, training=False)

        application.device = device
        application.batch_size = batch_size
        application.precision = precision

        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order)

    if arguments['apply']:

        model_pt = Path(arguments['<model.pt>'])
        model_pt = model_pt.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        # TODO. create README file in <output_dir>

        step = arguments['--step']
        if step is not None:
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = OverlapDetection.from_model_pt(
            model_pt, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step, subset=subset)
