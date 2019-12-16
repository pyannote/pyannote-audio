#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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
  pyannote-speaker-embedding validate [options] [--duration=<duration> --every=<epoch> --chronological --purity=<purity> --metric=<metric>] <train_dir> <database.task.protocol>
  pyannote-speaker-embedding apply [options] [--duration=<duration> --step=<step>] <validate_dir> <database.task.protocol>
  pyannote-speaker-embedding -h | --help
  pyannote-speaker-embedding --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<database.yml>  Path to pyannote.database configuration file.
  --subset=<subset>          Set subset (train|developement|test).
                             Defaults to "train" in "train" mode. Defaults to
                             "development" in "validate" mode. Defaults to
                             "test" in "apply" mode.
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
  --purity=<purity>          Target cluster purity [default: 0.9].
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --metric=<metric>          Use this metric (e.g. "cosine" or "euclidean") to
                             compare embeddings. Defaults to the metric defined
                             in "config.yml" configuration file.

"apply" mode:
  <validate_dir>             Path to the directory containing validation
                             results (i.e. the output of "validate" mode).
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network the architecture, and the approach used for training.

    ................... <experiment_dir>/config.yml ...................
    # train the network using triplet loss
    # see pyannote.audio.embedding.approaches for more details
    approach:
      name: TripletLoss
      params:
        metric: cosine    # embeddings are optimized for cosine metric
        clamp: positive   # triplet loss variant
        margin: 0.2       # triplet loss margin
        duration: 2       # sequence duration
        sampling: all     # triplet sampling strategy
        per_fold: 40      # number of speakers per fold
        per_label: 3      # number of sequences per speaker

    # use precomputed features (see feature extraction tutorial)
    feature_extraction:
      name: Precomputed
      params:
         root_dir: tutorials/feature-extraction

    # use the TristouNet architecture.
    # see pyannote.audio.embedding.models for more details
    architecture:
      name: TristouNet
      params:
        rnn: LSTM
        recurrent: [16]
        bidirectional: True
        pooling: sum
        linear: [16, 16]

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

    In practice, for each epoch, "validate" will
    * for speaker diarization protocols: tune the stopping criterion (distance
      threshold) of a "median-linkage" clustering using one average embedding
      for each reference speech turn and report both the best threshold and the
      corresponding purity/coverage F1 score to tensorboard
    * for speaker verification protocols: run the actual verification
      experiment and report the equal error rate to tensorboard,

"apply" mode:
    Use the "apply" mode to extract speaker embeddings on a sliding window.
    Resulting files can then be used in the following way:

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> embeddings = precomputed(first_test_file)
    >>> for window, embedding in embeddings:
    ...     # do something with embedding

"""

import torch
import numpy as np
from pathlib import Path
from docopt import docopt
from functools import partial
from typing import Optional

from .base import Application

from pyannote.core import Segment, Timeline, Annotation

from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerVerificationProtocol

from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage

from pyannote.core.utils.helper import get_class_by_name

from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import cdist
from pyannote.audio.features.precomputed import Precomputed

from pyannote.metrics.binary_classification import det_curve
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure

from pyannote.audio.embedding.extraction import SequenceEmbedding


class SpeakerEmbedding(Application):

    def __init__(self, experiment_dir, db_yml=None, training=False):

        super(SpeakerEmbedding, self).__init__(
            experiment_dir, db_yml=db_yml, training=training)

        # training approach
        Approach = get_class_by_name(
            self.config_['approach']['name'],
            default_module_name='pyannote.audio.embedding.approaches')
        self.task_ = Approach(
            **self.config_['approach'].get('params', {}))

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.embedding.models')
        params = self.config_['architecture'].get('params', {})
        self.get_model_ = partial(Architecture, **params)

        if hasattr(Architecture, 'get_frame_info'):
            self.frame_info_ = Architecture.get_frame_info(**params)
        else:
            self.frame_info_ = None

        if hasattr(Architecture, 'get_frame_crop'):
            self.frame_crop_ = Architecture.get_frame_crop(**params)
        else:
            self.frame_crop_ = None


    def validate_init(self, protocol_name, subset='development'):

        protocol = get_protocol(protocol_name)

        if isinstance(protocol, SpeakerVerificationProtocol):
            return self._validate_init_verification(protocol_name,
                                                    subset=subset)

        elif isinstance(protocol, SpeakerDiarizationProtocol):

            if self.duration is None:
                duration = getattr(self.task_, 'duration', None)
                if duration is None:
                    msg = ("Approach has no 'duration' defined. "
                           "Use '--duration' option to provide one.")
                    raise ValueError(msg)
                self.duration = duration

            return self._validate_init_diarization(protocol_name,
                                                   subset=subset)

        else:
            msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
                   'supported in "validation" mode.')
            raise ValueError(msg)

    def _validate_init_verification(self, protocol_name, subset='development'):
        return {}

    def _validate_init_diarization(self, protocol_name, subset='development'):
        return {}


    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        protocol = get_protocol(protocol_name)

        if isinstance(protocol, SpeakerVerificationProtocol):
            return self._validate_epoch_verification(
                epoch, protocol_name, subset=subset,
                validation_data=validation_data)

        elif isinstance(protocol, SpeakerDiarizationProtocol):
            return self._validate_epoch_diarization(
                epoch, protocol_name, subset=subset,
                validation_data=validation_data)

        else:
            msg = ('Only SpeakerVerification or SpeakerDiarization tasks are'
                   'supported in "validation" mode.')
            raise ValueError(msg)

    @staticmethod
    def get_hash(trial_file):
        uri = get_unique_identifier(trial_file)
        try_with = trial_file['try_with']
        if isinstance(try_with, Timeline):
            segments = tuple(try_with)
        else:
            segments = (try_with, )
        return hash((uri, segments))

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
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, min_duration=min_duration,
            batch_size=self.batch_size, device=self.device)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        y_true, y_pred, cache = [], [], {}

        for trial in getattr(protocol, '{0}_trial'.format(subset))():

            # compute embedding for file1
            file1 = trial['file1']
            hash1 = self.get_hash(file1)
            if hash1 in cache:
                emb1 = cache[hash1]
            else:
                emb1 = sequence_embedding.crop(file1, file1['try_with'])
                emb1 = np.mean(np.stack(emb1), axis=0, keepdims=True)
                cache[hash1] = emb1

            # compute embedding for file2
            file2 = trial['file2']
            hash2 = self.get_hash(file2)
            if hash2 in cache:
                emb2 = cache[hash2]
            else:
                emb2 = sequence_embedding.crop(file2, file2['try_with'])
                emb2 = np.mean(np.stack(emb2), axis=0, keepdims=True)
                cache[hash2] = emb2

            # compare embeddings
            distance = cdist(emb1, emb2, metric=self.metric)[0, 0]
            y_pred.append(distance)

            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred),
                                 distances=True)

        return {'metric': 'equal_error_rate',
                'minimize': True,
                'value': float(eer)}


    def _validate_epoch_diarization(self, epoch, protocol_name,
                                    subset='development',
                                    validation_data=None):
        """Perform a speaker diarization experiment using model at `epoch`

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
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=step, min_duration=min_duration,
            batch_size=self.batch_size, device=self.device)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        Z, t = dict(), dict()
        min_d, max_d = np.inf, -np.inf

        for current_file in getattr(protocol, subset)():

            uri = get_unique_identifier(current_file)
            uem = get_annotated(current_file)
            reference = current_file['annotation']

            X_, t_ = [], []
            embedding = sequence_embedding(current_file)
            for i, (turn, _) in enumerate(reference.itertracks()):

                # extract embedding for current speech turn. whenever possible,
                # only use those fully included in the speech turn ('strict')
                x_ = embedding.crop(turn, mode='strict')
                if len(x_) < 1:
                    x_ = embedding.crop(turn, mode='center')
                if len(x_) < 1:
                    x_ = embedding.crop(turn, mode='loose')
                if len(x_) < 1:
                    msg = (f'No embedding for {turn} in {uri:s}.')
                    raise ValueError(msg)

                # each speech turn is represented by its average embedding
                X_.append(np.mean(x_, axis=0))
                t_.append(turn)

            # apply hierarchical agglomerative clustering
            # all the way up to just one cluster (ie complete dendrogram)
            D = pdist(np.array(X_), metric=self.metric)
            min_d = min(np.min(D), min_d)
            max_d = max(np.max(D), max_d)

            Z[uri] = linkage(D, method='median')
            t[uri] = np.array(t_)

        def fun(threshold):

            metric = DiarizationPurityCoverageFMeasure(weighted=True)

            for current_file in getattr(protocol, subset)():

                uri = get_unique_identifier(current_file)
                uem = get_annotated(current_file)
                reference = current_file['annotation']

                clusters = fcluster(Z[uri], threshold, criterion='distance')

                hypothesis = Annotation(uri=uri)
                for (start_time, end_time), cluster in zip(t[uri], clusters):
                    hypothesis[Segment(start_time, end_time)] = cluster

                _ = metric(reference, hypothesis, uem=uem)

            purity, coverage, _ = metric.compute_metrics()

            return purity, coverage

        lower_threshold = min_d
        upper_threshold = max_d
        best_threshold = .5 * (lower_threshold + upper_threshold)
        best_coverage = 0.

        for _ in range(10):
            current_threshold = .5 * (lower_threshold + upper_threshold)
            purity, coverage = fun(current_threshold)

            if purity < self.purity:
                upper_threshold = current_threshold
            else:
                lower_threshold = current_threshold
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_threshold = current_threshold
        value = best_coverage if best_coverage else purity - self.purity
        return {'metric': f'coverage@{self.purity:.2f}purity',
                'minimize': False,
                'value': float(value)}


    def apply(self, protocol_name: str,
                    step: Optional[float] = None,
                    subset: Optional[str] = "test"):

        model = self.model_.to(self.device)
        model.eval()

        duration = self.duration
        if step is None:
            step = 0.25 * duration

        output_dir = Path(self.APPLY_DIR.format(
            validate_dir=self.validate_dir_,
            epoch=self.epoch_))

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize embedding extraction
        sequence_embedding = SequenceEmbedding(
            model=model,
            feature_extraction=self.feature_extraction_,
            duration=duration, step=step,
            batch_size=self.batch_size,
            device=self.device)

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

        for current_file in getattr(protocol, subset)():
            fX = sequence_embedding(current_file)
            precomputed.dump(current_file, fX)


def main():

    arguments = docopt(__doc__, version='Speaker embedding')

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

        application = SpeakerEmbedding(experiment_dir, db_yml=db_yml,
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

        purity = float(arguments['--purity'])

        application = SpeakerEmbedding.from_train_dir(
            train_dir, db_yml=db_yml, training=False)
        application.device = device
        application.purity = purity
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

        validate_dir = Path(arguments['<validate_dir>'])
        validate_dir = validate_dir.expanduser().resolve(strict=True)

        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        batch_size = int(arguments['--batch'])

        application = SpeakerEmbedding.from_validate_dir(
            validate_dir, db_yml=db_yml, training=False)
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

        application.apply(protocol_name, step=step, subset=subset)
