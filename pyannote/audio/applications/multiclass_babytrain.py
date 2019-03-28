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
# Marvin Lavechin - marvinlavechin@gmail.com

"""
Multi-class classifier BabyTrain

Usage:
  pyannote-multiclass-babytrain train [options] <experiment_dir> <database.task.protocol>
  pyannote-multiclass-babytrain validate [options] [--every=<epoch> --chronological --precision=<precision>] <label> <train_dir> <database.task.protocol>
  pyannote-multiclass-babytrain apply [options] [--step=<step>] <model.pt> <database.task.protocol> <output_dir>
  pyannote-multiclass-babytrain -h | --help
  pyannote-multiclass-babytrain --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "AMI.SpeakerDiarization.MixHeadset")
  --database=<database.yml>        Path to database configuration file.
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
  <label>                    Label to predict (speech, overlap, or change).
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).
  --precision=<precision>    Target detection precision [default: 0.8].

"apply" mode:
  <model.pt>                 Path to the pretrained model.
  --step=<step>              Sliding window step, in seconds.
                             Defaults to 25% of window duration.

Database configuration file <database.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process,
    the neural network architecture, and the task addressed.

    ................... <experiment_dir>/config.yml ...................
    # train the network for segmentation
    # see pyannote.audio.labeling.tasks for more details
    task:
       name: Segmentation
       params:
          duration: 3.2     # sub-sequence duration
          overlap: True     # train for overlap speech detection
          per_epoch: 1      # 1 day of audio per epoch
          batch_size: 32    # number of sub-sequences per batch

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
    Use the "apply" mode to extract segmentation raw scores.
    Resulting files can then be used in the following way:

    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('<output_dir>')

    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('<database.task.protocol>')
    >>> first_test_file = next(protocol.test())

    >>> from pyannote.audio.signal import Peak
    >>> peak_detection = Peak()

    >>> raw_scores = precomputed(first_test_file)
    >>> homogeneous_segments = peak_detection.apply(raw_scores, dimension=1)
"""

from functools import partial
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import scipy.optimize

import numpy as np
import torch
from docopt import docopt
from pyannote.database import get_annotated
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from pyannote.database import get_unique_identifier

from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature
from pyannote.core.utils.helper import get_class_by_name


from .base import Application


from pyannote.audio.features import Precomputed
from pyannote.audio.labeling.extraction import SequenceLabeling

from pyannote.audio.pipeline.speech_activity_detection \
    import SpeechActivityDetection as SpeechActivityDetectionPipeline

from pyannote.audio.pipeline.single_speaker_activity \
    import SingleSpeakerActivityDetection as SingleSpeakerActivityDetectionPipeline

from pyannote.metrics.detection import DetectionErrorRate

from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision



def validate_helper_func(current_file, pipeline=None, precision=None, recall=None, speaker_label=None):
    reference = current_file[speaker_label]
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file) # pipeline has been initialized with speaker_label, so that it can know which speaker needs to be assessed
    p = precision(reference, hypothesis, uem=uem)
    r = recall(reference, hypothesis, uem=uem)
    return p, r


class MulticlassBabyTrain(Application):

    def __init__(self, experiment_dir, db_yml=None, training=False):

        super().__init__(experiment_dir, db_yml=db_yml, training=training)
        # task
        Task = get_class_by_name(
            self.config_['task']['name'],
            default_module_name='pyannote.audio.labeling.tasks')
        self.task_ = Task(
            **self.config_['task'].get('params', {}))

        n_features = int(self.feature_extraction_.dimension)
        n_classes = self.task_.n_classes
        task_type = self.task_.task_type

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.labeling.models')
        self.model_ = Architecture(
            n_features, n_classes, task_type,
            **self.config_['architecture'].get('params', {}))

    def validate_init(self, protocol_name, subset='development'):

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)
        files = getattr(protocol, subset)()

        self.pool_ = mp.Pool(mp.cpu_count())

        # if features are already available on disk, return
        if isinstance(self.feature_extraction_, Precomputed):
            return list(files)

        # pre-compute features/overlap for each validation files
        validation_data = []
        for current_file in tqdm(files, desc='Feature extraction'):

            # precompute features
            if not isinstance(self.feature_extraction_, Precomputed):
                current_file['features'] = self.feature_extraction_(
                    current_file)
            # precompute "overlap" reference
            if self.task_.overlap:
                overlap = Timeline(uri=get_unique_identifier(current_file))
                reference = current_file['annotation']
                for track1, track2 in reference.co_iter(reference):
                    if track1 == track2:
                        continue
                    overlap.add(track1[0] & track2[0])
                current_file['overlap'] = overlap.to_annotation()

            # Extract subset relevant to the speaker whose speech performances need to be evaluated
            if self.label in ["KCHI", "CHI", "MAL", "FEM"]:
                reference = current_file['annotation']
                label_speech = reference.subset([self.label])
                current_file[self.label] = label_speech

            validation_data.append(current_file)

        return validation_data

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):
        if self.label == "overlap":
            func = getattr(self, f'validate_epoch_overlap')
            return func(epoch, protocol_name, subset=subset,
                        validation_data=validation_data)
        elif self.label in ["KCHI", "CHI", "MAL", "FEM"]:
            func = getattr(self, f'validate_epoch_speaker')
            return func(epoch, protocol_name, subset=subset,
                        validation_data=validation_data, speaker_role=self.label)
        elif self.label == "speech":
            func = getattr(self, f'validate_epoch_speech')
            return func(epoch, protocol_name, subset=subset,
                        validation_data=validation_data)
        else:
            raise ValueError("Argument label not recognized : %s " % self.label)

    def validate_epoch_speech(self, epoch, protocol_name, subset='development',
                              validation_data=None):
        """
        Validate the model for the SAD task
        """
        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration
        step = .25 * duration

        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

        for current_file in validation_data:
            scores = sequence_labeling(current_file)
            current_file['sad_scores'] = SlidingWindowFeature(
                scores.data[:, dimension].reshape(-1, 1),
                scores.sliding_window)

    def validate_epoch_speaker(self, epoch, protocol_name, subset='development',
                              validation_data=None, speaker_role='KCHI'):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration
        step = .25 * duration

        dimension = self.task_.labels.index(speaker_role)

        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)

        for current_file in validation_data:
            scores = sequence_labeling(current_file)
            current_file[speaker_role+'_scores'] = SlidingWindowFeature(
                scores.data[:, dimension].reshape(-1, 1),
                scores.sliding_window)

        # pipeline
        pipeline = SingleSpeakerActivityDetectionPipeline(speaker_label=self.label)

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
                               recall=recall,
                               speaker_label=self.label)
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

    def validate_epoch_overlap(self, epoch, protocol_name, subset='development',
                              validation_data=None):

        # load model for current epoch
        model = self.load_model(epoch).to(self.device)
        model.eval()

        # compute (and store) SAD scores
        duration = self.task_.duration
        step = .25 * duration

        dimension = self.task_.labels.index('overlap')

        sequence_labeling = SequenceLabeling(
            model=model, feature_extraction=self.feature_extraction_,
            duration=duration, step=.25 * duration, batch_size=self.batch_size,
            device=self.device)
        for current_file in validation_data:
            scores = sequence_labeling(current_file)
            current_file['sad_scores'] = SlidingWindowFeature(
                scores.data[:, dimension].reshape(-1, 1),
                scores.sliding_window)

        # pipeline
        pipeline = SpeechActivityDetectionPipeline()

        def fun(threshold):
            pipeline.instantiate({'onset': threshold,
                                  'offset': threshold,
                                  'min_duration_on': 0.,
                                  'min_duration_off': 0.,
                                  'pad_onset': 0.,
                                  'pad_offset': 0.})
            metric = DetectionErrorRate(parallel=True)
            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               metric=metric,
                               reference='overlap')
            _ = self.pool_.map(validate, validation_data)

            return abs(metric)

        res = scipy.optimize.minimize_scalar(
            fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

        threshold = res.x.item()
        return {'metric': 'detection_error_rate',
                'minimize': True,
                'value': res.fun,
                'pipeline': pipeline.instantiate({'onset': threshold,
                                                  'offset': threshold,
                                                  'min_duration_on': 0.,
                                                  'min_duration_off': 0.,
                                                  'pad_onset': 0.,
                                                  'pad_offset': 0.})}

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
        labels = self.task_.labels

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=n_classes,
            labels=labels)

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
    arguments = docopt(__doc__, version='MulticlassBabyTrain')
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

        application = MulticlassBabyTrain(experiment_dir, db_yml=db_yml,
                                             training=True)
        application.device = device
        application.train(protocol_name, subset=subset,
                          restart=restart, epochs=epochs)

    if arguments['validate']:
        label = arguments['<label>']
        precision = float(arguments['--precision'])
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

        application = MulticlassBabyTrain.from_train_dir(
            train_dir, db_yml=db_yml, training=False)

        application.device = device
        application.batch_size = batch_size
        application.label = label
        application.precision = precision

        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             in_order=in_order, task=label)

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

        application = MulticlassBabyTrain.from_model_pt(
            model_pt, db_yml=db_yml, training=False)
        application.device = device
        application.batch_size = batch_size
        application.apply(protocol_name, output_dir, step=step, subset=subset)