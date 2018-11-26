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
  pyannote-speech-detection test [options] [--dump=<output_dir>] <train_dir> <params.yml> <database.task.protocol>
  pyannote-speech-detection -h | --help
  pyannote-speech-detection --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
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

"apply" mode:
  <model.pt>                 Path to the pretrained model.
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

"test" mode:
    <train_dir>           Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
    <params.yml>          Path to best hyper-parameters found in validation.
    --dump=<output_dir>   Directory where to store the detection output.

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
          per_epoch: 1      # 1 day of audio per epoch
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

"test" mode:
    Use the "test" mode to run the best model found in validation on the test dataset,
    computing a set o standard metrics. Optionally, dump the results to a file
"""

import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
import yaml
from docopt import docopt
from pyannote.core import SlidingWindowFeature
from pyannote.database import FileFinder
from pyannote.database import get_annotated
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier
from pyannote.metrics.detection import DetectionErrorRate
from tqdm import tqdm

from pyannote.audio.features import Precomputed
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.signal import Binarize
from pyannote.audio.util import get_class_by_name, mkdir_p
from pyannote.core.utils.helper import get_class_by_name
from functools import partial
import multiprocessing as mp
from pyannote.audio.pipeline import SpeechActivityDetection \
                             as SpeechActivityDetectionPipeline

def validate_helper_func(current_file, pipeline=None, metric=None):
    reference = current_file['annotation']
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file)
    return metric(reference, hypothesis, uem=uem)

class SpeechActivityDetection(Application):

    def __init__(self, experiment_dir, db_yml=None, training=False):

        super(SpeechActivityDetection, self).__init__(
            experiment_dir, db_yml=db_yml, training=training)

        # task
        Task = get_class_by_name(
            self.config_['task']['name'],
            default_module_name='pyannote.audio.labeling.tasks')
        self.task_ = Task(
            **self.config_['task'].get('params', {}))

        n_features = int(self.feature_extraction_.dimension)
        n_classes = self.task_.n_classes

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.labeling.models')
        self.model_ = Architecture(
            n_features, n_classes,
            **self.config_['architecture'].get('params', {}))

    def validate_init(self, protocol_name, subset='development'):

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        self.pool_ = mp.Pool(mp.cpu_count())

        if isinstance(self.feature_extraction_, Precomputed):
            return list(files)

        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'):
            current_file['features'] = self.feature_extraction_(current_file)
            validation_data.append(current_file)

        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration
        step = .25 * duration
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)
        for current_file in validation_data:
            uri = get_unique_identifier(current_file)
            current_file['sad_scores'] = sequence_labeling(current_file)

        # pipeline
        pipeline = SpeechActivityDetectionPipeline()
        pipeline.freeze({'min_duration_on': 0.,
                         'min_duration_off': 0.,
                         'pad_onset': 0.,
                         'pad_offset': 0.})

        def fun(threshold):
            pipeline.with_params({'onset': threshold, 'offset': threshold})
            metric = DetectionErrorRate(parallel=True)
            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               metric=metric)
            _ = self.pool_.map(validate, validation_data)

            return abs(metric)

        res = scipy.optimize.minimize_scalar(
            fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

        threshold = res.x
        return {'metric': 'detection_error_rate',
                'minimize': True,
                'value': res.fun,
                'pipeline': pipeline.with_params({'onset': threshold,
                                                        'offset': threshold})}

    def test(self, params, protocol_name, output_dir=None):

        epoch = params['epoch']
        threshold = params['sad_onset']

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        test_files = getattr(protocol, 'test')()

        duration = self.task_.duration
        step = .25 * duration
        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, batch_size=self.batch_size,
            device=self.device)

        # extract predictions for all files.
        predictions = {}
        for current_file in test_files:
            uri = get_unique_identifier(current_file)
            scores = sequence_labeling(current_file)

            if model.logsoftmax:
                scores = SlidingWindowFeature(
                    1. - np.exp(scores.data[:, 0]),
                    scores.sliding_window)
            else:
                scores = SlidingWindowFeature(
                    1. - scores.data[:, 0],
                    scores.sliding_window)

            predictions[uri] = scores

        # apply threshold from parameters to obtain annotations
        binarizer = Binarize(onset=threshold,
                             offset=threshold,
                             log_scale=False)

        # dump results to file
        if output_dir:

            mkdir_p(output_dir)

            path = Path(output_dir) / f'{protocol_name}.test.txt'

            with open(path, mode='w') as fp:

                for uri, prediction in predictions.items():

                    hypothesis = binarizer.apply(
                        prediction, dimension=0).to_annotation()


                    for s, t, l in hypothesis.itertracks(yield_label=True):
                        fp.write(f'{uri} {s.start:.3f} {s.end:.3f} V\n')

        # compute metric
        metric = DetectionErrorRate(parallel=True)
        validate = partial(validate_helper_func,
                           predictions=predictions,
                           binarizer=binarizer,
                           metric=metric)

        test_files = getattr(protocol, 'test')()
        self.pool_ = mp.Pool(mp.cpu_count())
        _ = self.pool_.map(validate, test_files)

        return abs(metric)

    def apply(self, protocol_name, output_dir, step=None, subset=None):

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
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

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

        if subset is None:
            files = FileFinder.protocol_file_iter(protocol,
                                                  extra_keys=['audio'])
        else:
            files = getattr(protocol, subset)()

        for current_file in files:
            fX = sequence_labeling(current_file)
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

        application = SpeechActivityDetection(experiment_dir, db_yml=db_yml,
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

        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
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

        application = SpeechActivityDetection.from_model_pt(
            model_pt, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step, subset=subset)

    if arguments['test']:

        batch_size = int(arguments['--batch'])
        output_dir = arguments['--dump']

        train_dir = Path(arguments['<train_dir>'])
        train_dir = train_dir.expanduser().resolve(strict=True)

        # params
        params_yml = Path(arguments['<params.yml>'])
        params_yml = params_yml.expanduser().resolve(strict=True)

        with open(params_yml, 'r') as f:
            params = yaml.load(f)

        application = SpeechActivityDetection.from_train_dir(
            train_dir, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size

        metric = application.test(params, protocol_name, output_dir=output_dir)

        print(f'Test EER: {100*metric: .3f}%')
