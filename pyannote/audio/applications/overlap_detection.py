#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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
Overlap speech detection

Usage:
  pyannote-overlap-detection train [options] <experiment_dir> <database.task.protocol>
  pyannote-overlap-detection validate [options] [--every=<epoch> --chronological --precision=<precision>] <train_dir> <database.task.protocol>
  pyannote-overlap-detection apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-overlap-detection -h | --help
  pyannote-overlap-detection --version

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
  --precision=<precision>    Target precision [default: 0.9].

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
       name: OverlapDetection
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

"apply" mode
    Finally, one can apply overlap speech detection using "apply" mode.
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
from os.path import expanduser
from pyannote.core import Timeline
from pyannote.database import get_protocol
from pyannote.audio.signal import Binarize
from pyannote.database import get_annotated
from pyannote.core import SlidingWindowFeature
from pyannote.audio.features import Precomputed
from pyannote.database import get_unique_identifier
from .speech_detection import SpeechActivityDetection
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision
from pyannote.audio.labeling.extraction import SequenceLabeling


class OverlapDetection(SpeechActivityDetection):

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        target_precision = self.precision

        # load model for current epoch
        model = self.load_model(epoch)
        if self.gpu:
            model = model.cuda()
        model.eval()

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

        predictions = {}
        references = {}

        file_generator = getattr(protocol, subset)()
        for current_file in file_generator:
            uri = get_unique_identifier(current_file)

            # build overlap reference
            reference = Timeline(uri=uri)
            annotation = current_file['annotation']
            for track1, track2 in annotation.co_iter(annotation):
                if track1 == track2:
                    continue
                reference.add(track1[0] & track2[0])
            references[uri] = reference.to_annotation()

            # extract overlap scores
            scores = sequence_labeling.apply(current_file)

            if model.logsoftmax:
                scores = SlidingWindowFeature(
                    np.exp(scores.data[:, 2]), scores.sliding_window)
            else:
                scores = SlidingWindowFeature(
                    scores.data[:, 2], scores.sliding_window)

            predictions[uri] = scores

        # dichotomic search to find threshold that maximizes recall
        # while having at least `target_precision`

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        for _ in range(10):
            current_alpha = .5 * (lower_alpha + upper_alpha)
            binarizer = Binarize(onset=current_alpha,
                                 offset=current_alpha,
                                 log_scale=False)

            precision = DetectionPrecision()
            recall = DetectionRecall()

            for current_file in getattr(protocol, subset)():
                uri = get_unique_identifier(current_file)
                reference = references[uri]
                hypothesis = binarizer.apply(predictions[uri], dimension=0)
                hypothesis = hypothesis.to_annotation()
                uem = get_annotated(current_file)
                _ = precision(reference, hypothesis, uem=uem)
                _ = recall(reference, hypothesis, uem=uem)

            if abs(precision) < target_precision:
                # precision is not high enough: try higher thresholds
                lower_alpha = current_alpha
            else:
                upper_alpha = current_alpha
                r = abs(recall)
                if r > best_recall:
                    best_recall = r
                    best_alpha = current_alpha

        metric_name = f'RecallAt{target_precision:.2f}Precision'
        return {
            metric_name: {'minimize': False, 'value': best_recall},
            'binarize/threshold': {'minimize': 'NA',
                                              'value': best_alpha}}

def main():

    arguments = docopt(__doc__, version='Overlapping speech detection')

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

        application = OverlapDetection(experiment_dir, db_yml=db_yml)
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

        precision = float(arguments['--precision'])

        application = OverlapDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.gpu = gpu
        application.batch_size = batch_size
        application.precision = precision
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

        application = OverlapDetection.from_model_pt(
            model_pt, db_yml=db_yml)
        application.gpu = gpu
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step)
