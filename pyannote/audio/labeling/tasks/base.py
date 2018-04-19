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

import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from pyannote.metrics.binary_classification import det_curve
from pyannote.database import get_unique_identifier
from pyannote.database import get_label_identifier
from pyannote.database import get_annotated
from pyannote.audio.util import to_numpy
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature

from pyannote.generators.batch import batchify
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.fragment import SlidingSegments

from collections import deque

from pyannote.audio.train import Trainer


class LabelingTaskGenerator(object):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    shuffle : bool, optional
        Shuffle exhaustive samples. Defaults to False.
    """

    def __init__(self, precomputed, duration=3.2, batch_size=32,
                 per_epoch=3600, parallel=1, exhaustive=False, shuffle=False):

        super(LabelingTaskGenerator, self).__init__()

        self.precomputed = precomputed
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.exhaustive = exhaustive
        self.shuffle = shuffle

    def initialize(self, protocol, subset='train'):
        """Gather the following information about the training subset:

        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        databases_ : list
            Sorted list of (unique) databases in protocol.

        labels_ : list
            Sorted list of (unique) lables in protocol.
        """

        self.data_ = {}
        labels, databases = set(), set()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of database
            database = current_file['database']
            databases.add(database)

            # keep track of unique labels
            for label in current_file['annotation'].labels():
                label = get_label_identifier(label, current_file)
                labels.add(label)

            annotated = get_annotated(current_file)

            if not self.precomputed.use_memmap:
                msg = ('Loading all precomputed features in memory. '
                       'Set "use_memmap" to True if you run out of memory.')
                warnings.warn(msg)

            segments = [s for s in annotated if s.duration > self.duration]

            # corner case where no segment is long enough
            # and we removed them all...
            if not segments:
                continue

            # total duration of label in current_file (after removal of
            # short segments).
            duration = sum(s.duration for s in segments)

            # store all these in data_ dictionary
            datum = {'segments': segments,
                     'duration': duration,
                     'current_file': current_file}
            uri = get_unique_identifier(current_file)
            self.data_[uri] = datum

        self.databases_ = sorted(databases)
        self.labels_ = sorted(labels)

        sliding_window = self.precomputed.sliding_window()
        for current_file in getattr(protocol, subset)():
            y, _ = to_numpy(current_file, self.precomputed,
                            labels=self.labels_)
            uri = get_unique_identifier(current_file)
            self.data_[uri]['y'] = SlidingWindowFeature(
                self.postprocess_y(y), sliding_window)

    def postprocess_y(self, Y):
        """This function does nothing but return its input.
        It should be overriden by subclasses."""
        return Y

    def samples(self):
        if self.exhaustive:
            return self.random_samples()
        else:
            return self.sliding_samples()

    def random_samples(self):
        """Random samples

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        while True:

            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]

            datum = self.data_[uri]
            current_file = datum['current_file']

            # choose one segment at random with probability
            # proportional to its duration
            segment = next(random_segment(datum['segments'], weighted=True))

            # choose fixed-duration subsegment at random
            sequence = next(random_subsegment(segment, self.duration))

            X = self.precomputed.crop(current_file,
                                      sequence, mode='center',
                                      fixed=self.duration)

            y = datum['y'].crop(sequence, mode='center', fixed=self.duration)

            yield {'X': X, 'y': np.squeeze(y)}

    def sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        sliding_segments = SlidingSegments(duration=self.duration,
                                           step=self.duration,
                                           source='annotated')

        while True:

            random.shuffle(uris)

            # loop on all files
            for uri in uris:
                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum['current_file'])

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                annotated = Timeline(
                    [Segment(s.start + random.random() * self.duration,
                             s.end) for s in get_annotated(current_file)])
                current_file['annotated'] = annotated

                if self.shuffle:
                    samples = []

                for sequence in sliding_segments.from_file(current_file):

                    X = self.precomputed.crop(current_file,
                                              sequence, mode='center',
                                              fixed=self.duration)

                    y = datum['y'].crop(sequence, mode='center',
                                        fixed=self.duration)

                    sample = {'X': X, 'y': np.squeeze(y)}

                    if self.shuffle:
                        samples.append(sample)
                    else:
                        yield sample

                if self.shuffle:
                    random.shuffle(samples)
                    for sample in samples:
                        yield sample

    @property
    def signature(self):
        """Generator signature"""
        return {'X': {'type': 'ndarray'},
                'y': {'type': 'ndarray'}}

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(self.per_epoch / duration_per_batch))

    @property
    def labels(self):
        return list(self.labels_)

    def __call__(self, protocol, subset='train'):
        """(Parallelized) batch generator"""

        # pre-load useful information about protocol once and for all
        self.initialize(protocol, subset=subset)

        # number of batches needed to complete an epoch
        batches_per_epoch = self.batches_per_epoch

        generators = []

        if self.parallel:
            for _ in range(self.parallel):

                # initialize one sample generator
                samples = self.samples()

                # batchify it and make sure at least
                # `batches_per_epoch` batches are prefetched.
                batches = batchify(samples, self.signature,
                                   batch_size=self.batch_size,
                                   prefetch=batches_per_epoch)

                # add batch generator to the list of (background) generators
                generators.append(batches)
        else:

            # initialize one sample generator
            samples = self.samples()

            # batchify it without prefetching
            batches = batchify(samples, self.signature,
                               batch_size=self.batch_size, prefetch=0)

            # add it to the list of generators
            # NOTE: this list will only contain one generator
            generators.append(batches)

        # loop on (background) generators indefinitely
        while True:
            for batches in generators:
                # yield `batches_per_epoch` batches from current generator
                # so that each epoch is covered by exactly one generator
                for _ in range(batches_per_epoch):
                    yield next(batches)


class LabelingTask(Trainer):
    """Base class for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    optimizer : {'sgd', 'rmsprop', 'adam'}
        Defaults to 'sgd'.
    learning_rate : float, optional
        Learning rate. Defaults to 0.01.
    enable_backtrack : bool, optional
        Defaults to True.
    """

    def __init__(self, duration=3.2, batch_size=32, per_epoch=3600,
                 parallel=1, optimizer='sgd', learning_rate=1e-2,
                 enable_backtrack=True):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.enable_backtrack = enable_backtrack

    def get_batch_generator(self, precomputed):
        """This method should be overriden by subclass

        Parameters
        ----------
        precomputed : `pyannote.audio.features.Precomputed`

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        return LabelingTaskGenerator(
            precomputed, duration=self.duration, per_epoch=self.per_epoch,
            batch_size=self.batch_size, parallel=self.parallel)

    @property
    def n_classes(self):
        """Number of classes"""
        msg = 'LabelingTask subclass must define `n_classes` property.'
        raise NotImplementedError(msg)

    def on_train_start(self):

        if self.model_.n_classes != self.n_classes:
            raise ValueError('n_classes mismatch')

        self.loss_func_ = self.model_.get_loss()

        self.log_y_pred_ = deque([], maxlen=self.batches_per_epoch_)
        self.log_y_true_ = deque([], maxlen=self.batches_per_epoch_)

    def process_batch(self, batch):

        X, y = batch['X'], batch['y']
        if not getattr(self.model_, 'batch_first', True):
            X = np.rollaxis(X, 0, 2)
            y = y.T
        X = np.array(X, dtype=np.float32)
        X = Variable(torch.from_numpy(X))
        y = Variable(torch.from_numpy(y))

        if self.gpu_:
            X = X.cuda()
            y = y.cuda()

        fX = self.model_(X)

        losses = self.loss_func_(fX.view((-1, self.n_classes)),
                           y.contiguous().view((-1, )))

        if self.detailed_log_:
            self.log_y_pred_.append(self.to_numpy(fX))
            self.log_y_true_.append(self.to_numpy(y))

        return torch.mean(losses)

    def on_epoch_end(self, epoch):

        if not self.detailed_log_:
            return

        log_y_pred = np.hstack(self.log_y_pred_)
        log_y_true = np.hstack(self.log_y_true_)
        log_y_pred = log_y_pred.reshape((-1, self.n_classes))
        log_y_true = log_y_true.reshape((-1, ))
        for k in range(self.n_classes):
            _, _, _, eer = det_curve(log_y_true == k,
                                     log_y_pred[:, k])
            self.writer_.add_scalar(f'train/estimate/eer/{k}',
                eer, global_step=epoch)
