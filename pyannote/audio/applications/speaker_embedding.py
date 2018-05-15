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
Speaker embedding

Usage:
  pyannote-speaker-embedding train [options] <experiment_dir> <database.task.protocol>
  pyannote-speaker-embedding validate [options] [--duration=<duration> --every=<epoch> --chronological --turn --metric=<metric>] <train_dir> <database.task.protocol>
  pyannote-speaker-embedding apply [options] [--duration=<duration> --step=<step> --internal --normalize] <model.pt> <database.task.protocol> <output_dir>
  pyannote-speaker-embedding -h | --help
  pyannote-speaker-embedding --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
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
  --duration=<duration>      {Validate|apply} using subsequences with that
                             duration. Defaults to embedding fixed duration
                             when available.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  --chronological            Force validation in chronological order.
  --turn                     Perform same/different validation at speech turn
                             level. Default is to use fixed duration segments.
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --metric=<metric>          Use this metric (e.g. "cosine" or "euclidean") to
                             compare embeddings. Defaults to the metric defined
                             in "config.yml" configuration file.

"apply" mode:
  <model.pt>                 Path to the pretrained model.
  --internal                 Extract internal embeddings.
  --normalize                Extract normalized embeddings.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence labeling (0 vs. 1, non-speech vs. speech), the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    both training and testing.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: Precomputed
       params:
          root_dir: /vol/work1/bredin/feature_extraction/mfcc

    architecture:
       name: ClopiNet
       params:
         rnn: LSTM
         recurrent: [64, 64, 64]
         bidirectional: True
         linear: []
         weighted: False

    approach:
       name: TripletLoss
       params:
         metric: cosine
         margin: 0.1
         clamp: positive
         duration: 3.2
         sampling: all
         per_fold: 100
         per_label: 3
    ...................................................................

"train" mode:
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

"validate" mode:
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results:

        <train_dir>/validate/<database.task.protocol>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).
"""

import torch
import itertools
import numpy as np
from docopt import docopt
from .base import Application
from os.path import expanduser
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.audio.embedding.utils import pdist
from pyannote.audio.embedding.utils import cdist
from pyannote.database import get_unique_identifier
from pyannote.audio.features.utils import Precomputed
from pyannote.metrics.binary_classification import det_curve
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.generators.speaker import SpeechTurnSubSegmentGenerator


class SpeakerEmbedding(Application):

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeakerEmbedding, self).__init__(
            experiment_dir, db_yml=db_yml)

        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.embedding.models',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        self.model_ = Architecture(
            int(self.feature_extraction_.dimension()),
            **self.config_['architecture'].get('params', {}))

        approach_name = self.config_['approach']['name']
        approaches = __import__('pyannote.audio.embedding.approaches',
                                fromlist=[approach_name])
        Approach = getattr(approaches, approach_name)
        self.task_ = Approach(
            **self.config_['approach'].get('params', {}))


    def validate_init(self, protocol_name, subset='development'):

        task = protocol_name.split('.')[1]
        if task == 'SpeakerVerification':
            return self._validate_init_verification(protocol_name,
                                                    subset=subset)

        elif task == 'SpeakerDiarization':

            if self.duration is None:
                duration = getattr(self.task_, 'duration', None)
                if duration is None:
                    msg = ("Approach has no 'duration' defined. "
                           "Use '--duration' option to provide one.")
                    raise ValueError(msg)
                self.duration = duration

            if self.turn:
                return self._validate_init_turn(protocol_name,
                                                subset=subset)
            else:
                return self._validate_init_segment(protocol_name,
                                                   subset=subset)

        else:
            msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
                   'supported in "validation" mode.')
            raise ValueError(msg)

    def _validate_init_verification(self, protocol_name, subset='development'):
        return {}

    def _validate_init_turn(self, protocol_name, subset='development'):

        np.random.seed(1337)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        batch_generator = SpeechTurnSubSegmentGenerator(
            self.feature_extraction_, self.duration,
            per_label=10, per_turn=5)
        batch = next(batch_generator(protocol, subset=subset))

        X = np.stack(batch['X'])
        y = np.stack(batch['y'])
        z = np.stack(batch['z'])

        # get list of labels from list of repeated labels:
        # z 0 0 0 1 1 1 2 2 2 2 3 3 3 3
        # y A A A A A A B B B B B B B B
        # becomes
        # z 0 0 0 1 1 1 2 2 2 2 3 3 3 3
        # y A B
        yz = np.vstack([y, z]).T
        y = []
        for _, yz_ in itertools.groupby(yz, lambda t: t[1]):
            yz_ = np.stack(yz_)
            y.append(yz_[0, 0])
        y = np.array(y).reshape((-1, 1))

        # precompute same/different groundtruth
        y = pdist(y, metric='equal')

        return {'X': X, 'y': y, 'z': z}

    def _validate_init_segment(self, protocol_name, subset='development'):

        np.random.seed(1337)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        batch_generator = SpeechSegmentGenerator(
            self.feature_extraction_, per_label=10, duration=self.duration)
        batch = next(batch_generator(protocol, subset=subset))

        X = np.stack(batch['X'])
        y = np.stack(batch['y']).reshape((-1, 1))

        # precompute same/different groundtruth
        y = pdist(y, metric='equal')
        return {'X': X, 'y': y}

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        task = protocol_name.split('.')[1]
        if task == 'SpeakerVerification':
            return self._validate_epoch_verification(
                epoch, protocol_name, subset=subset,
                validation_data=validation_data)

        elif task == 'SpeakerDiarization':
            if self.turn:
                return self._validate_epoch_turn(
                    epoch, protocol_name, subset=subset,
                    validation_data=validation_data)
            else:
                return self._validate_epoch_segment(
                    epoch, protocol_name, subset=subset,
                    validation_data=validation_data)

        else:
            msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
                   'supported in "validation" mode.')
            raise ValueError(msg)

    def _validate_epoch_verification(self, epoch, protocol_name,
                                     subset='development',
                                     validation_data=None):
        """Perform a speaker verification experiment using model at `epoch`

        Parameters
        ----------
        epoch : int
            Epoch to validate.
        protocol_name : str
            Name of speaker verification protocol
        subset : {'train', 'development', 'test'}, optional
            Name of subset.
        validation_data : provided by `validate_init`

        Returns
        -------
        metrics : dict
        """


        # load current model
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # use final representation (not internal ones)
        model.internal = False

        
        # use user-provided --duration when available
        # otherwise use 'duration' used for training
        if self.duration is None:
            duration = self.task_.duration
        else:
            duration = self.duration
        min_duration = None

        # if 'duration' is still None, it means that
        # network was trained with variable lengths
        if duration is None:
            duration = self.task_.max_duration
            min_duration = self.task_.min_duration

        step = .5 * duration

        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_embedding = SequenceEmbedding(
            model, self.feature_extraction_, duration=duration,
            step=step, min_duration=min_duration,
            batch_size=self.batch_size, device=self.device)

        metrics = {}
        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        enrolment_models, enrolment_khashes = {}, {}
        enrolments = getattr(protocol, '{0}_enrolment'.format(subset))()
        for i, enrolment in enumerate(enrolments):
            data = sequence_embedding.apply(enrolment,
                                            crop=enrolment['enrol_with'])
            model_id = enrolment['model_id']
            model = np.mean(np.stack(data), axis=0, keepdims=True)
            enrolment_models[model_id] = model

            # in some specific speaker verification protocols,
            # enrolment data may be  used later as trial data.
            # therefore, we cache information about enrolment data
            # to speed things up by reusing the enrolment as trial
            h = hash((get_unique_identifier(enrolment),
                      tuple(enrolment['enrol_with'])))
            enrolment_khashes[h] = model_id

        trial_models = {}
        trials = getattr(protocol, '{0}_trial'.format(subset))()
        y_true, y_pred = [], []
        for i, trial in enumerate(trials):
            model_id = trial['model_id']

            h = hash((get_unique_identifier(trial),
                      tuple(trial['try_with'])))

            # re-use enrolment model whenever possible
            if h in enrolment_khashes:
                model = enrolment_models[enrolment_khashes[h]]

            # re-use trial model whenever possible
            elif h in trial_models:
                model = trial_models[h]

            else:
                data = sequence_embedding.apply(trial, crop=trial['try_with'])
                model = np.mean(data, axis=0, keepdims=True)
                # cache trial model for later re-use
                trial_models[h] = model

            distance = cdist(enrolment_models[model_id], model,
                             metric=self.metric)[0, 0]
            y_pred.append(distance)
            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred),
                                 distances=True)
        metrics['EER'] = {'minimize': True, 'value': eer}

        return metrics

    def _validate_epoch_segment(self, epoch, protocol_name,
                                subset='development',
                                validation_data=None):

        model = self.load_model(epoch).to(self.device)
        model.eval()

        sequence_embedding = SequenceEmbedding(
            model, self.feature_extraction_,
            batch_size=self.batch_size, device=self.device)


        fX = sequence_embedding.apply(validation_data['X'])
        y_pred = pdist(fX, metric=self.metric)
        _, _, _, eer = det_curve(validation_data['y'], y_pred,
                                 distances=True)

        return {'EER.{0:g}s'.format(self.duration): {'minimize': True,
                                                'value': eer}}

    def _validate_epoch_turn(self, epoch, protocol_name,
                             subset='development',
                             validation_data=None):

        model = self.load_model(epoch).to(self.device)
        model.eval()

        sequence_embedding = SequenceEmbedding(
            model, self.feature_extraction_,
            batch_size=self.batch_size, device=self.device)

        fX = sequence_embedding.apply(validation_data['X'])

        z = validation_data['z']

        # iterate over segments, speech turn by speech turn

        fX_avg = []
        nz = np.vstack([np.arange(len(z)), z]).T
        for _, nz_ in itertools.groupby(nz, lambda t: t[1]):

            # (n, 2) numpy array where
            # * n is the number of segments in current speech turn
            # * dim #0 is the index of segment in original batch
            # * dim #1 is the index of speech turn (used for grouping)
            nz_ = np.stack(nz_)

            # compute (and stack) average embedding over all segments
            # of current speech turn
            indices = nz_[:, 0]

            fX_avg.append(np.mean(fX[indices], axis=0))

        fX = np.vstack(fX_avg)
        y_pred = pdist(fX, metric=self.metric)
        _, _, _, eer = det_curve(validation_data['y'], y_pred,
                                 distances=True)
        metrics = {}
        metrics['EER.turn'] = {'minimize': True, 'value': eer}
        return metrics

    def apply(self, protocol_name, output_dir, step=None,
              internal=False, normalize=False):

        model = self.model_.to(self.device)
        model.eval()

        if internal is not None:
            model.internal = internal
        if normalize is not None:
            model.normalize = normalize

        duration = self.duration
        if step is None:
            step = 0.5 * duration

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_embedding = SequenceEmbedding(
            model, self.feature_extraction_, duration=duration,
            step=step, batch_size=self.batch_size, device=self.device)
        sliding_window = sequence_embedding.sliding_window
        dimension = sequence_embedding.dimension

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=dimension)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for current_file in FileFinder.protocol_file_iter(
            protocol, extra_keys=['audio']):

            fX = sequence_embedding.apply(current_file)
            precomputed.dump(current_file, fX)

def main():

    arguments = docopt(__doc__, version='Speaker embedding')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']
    device = torch.device('cuda') if gpu else torch.device('cpu')


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

        application = SpeakerEmbedding(experiment_dir, db_yml=db_yml)
        application.device = device
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

        # validate at speech turn level
        turn = arguments['--turn']

        # batch size
        batch_size = int(arguments['--batch'])

        application = SpeakerEmbedding.from_train_dir(
            train_dir, db_yml=db_yml)
        application.device = device
        application.turn = turn
        application.batch_size = batch_size

        metric = arguments['--metric']
        if metric is None:
            metric = getattr(application.task_, 'metric', None)
            if metric is None:
                msg = ("Approach has no 'metric' defined. "
                       "Use '--metric' option to provide one.")
                raise ValueError(msg)
        application.metric = metric

        duration = arguments['--duration']
        if duration is not None:
            duration = float(duration)
        application.duration = duration

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

        internal = arguments['--internal']
        normalize = arguments['--normalize']
        batch_size = int(arguments['--batch'])

        application = SpeakerEmbedding.from_model_pt(
            model_pt, db_yml=db_yml)
        application.device = device
        application.batch_size = batch_size

        duration = arguments['--duration']
        if duration is None:
            duration = getattr(application.task_, 'duration', None)
            if duration is None:
                msg = ("Approach has no 'duration' defined. "
                       "Use '--duration' option to provide one.")
                raise ValueError(msg)
        else:
            duration = float(duration)
        application.duration = duration

        application.apply(protocol_name, output_dir, step=step,
                          internal=internal, normalize=normalize)
