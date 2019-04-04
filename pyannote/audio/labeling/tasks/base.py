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

import warnings
import torch
import numpy as np
from tqdm import tqdm
from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.audio.features import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature

from pyannote.generators.batch import batchify
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.fragment import SlidingSegments

from pyannote.audio.train.trainer import Trainer

from .. import TASK_CLASSIFICATION
from .. import TASK_MULTI_LABEL_CLASSIFICATION
from .. import TASK_REGRESSION

import torch.nn.functional as F


class LabelingTaskGenerator(object):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
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

    def __init__(self, feature_extraction, protocol, subset='train',
                 duration=3.2, batch_size=32, per_epoch=1, parallel=1,
                 exhaustive=False, shuffle=False):

        super(LabelingTaskGenerator, self).__init__()

        self.feature_extraction = feature_extraction
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.exhaustive = exhaustive
        self.shuffle = shuffle

        self._load_metadata(protocol, subset=subset)

    def postprocess_y(self, Y):
        """This function does nothing but return its input.
        It should be overriden by subclasses."""
        return Y

    def initialize_y(self, current_file):
        """Precompute y for the whole file

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        y : `SlidingWindowFeature`
            Precomputed y for the whole file
        """
        y, _ = one_hot_encoding(current_file['annotation'],
                                get_annotated(current_file),
                                self.feature_extraction.sliding_window,
                                labels=self.original_classes_, mode='center')

        return SlidingWindowFeature(self.postprocess_y(y.data),
                                    y.sliding_window)

    def crop_y(self, y, segment):
        """Extract y for specified segment

        Parameters
        ----------
        y : `pyannote.core.SlidingWindowFeature`
            Output of `initialize_y` above.
        segment : `pyannote.core.Segment`
            Segment for which to obtain y.

        Returns
        -------
        cropped_y :  (n_samples, ) or (n_samples, dim) `np.ndarray`
            y for specified
        """

        return np.squeeze(y.crop(segment, mode='center', fixed=self.duration))

    def _load_metadata(self, protocol, subset='train'):
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

            # ensure annotation/annotated are cropped to actual file duration
            support = Segment(start=0, end=get_audio_duration(current_file))
            current_file['annotated'] = get_annotated(current_file).crop(
                support, mode='intersection')
            current_file['annotation'] = current_file['annotation'].crop(
                support, mode='intersection')

            # keep track of database
            database = current_file['database']
            databases.add(database)

            # keep track of unique labels
            labels.update(current_file['annotation'].labels())

            if isinstance(self.feature_extraction, Precomputed) and \
               not self.feature_extraction.use_memmap:
                msg = ('Loading all precomputed features in memory. '
                       'Set "use_memmap" to True if you run out of memory.')
                warnings.warn(msg)

            segments = [s for s in current_file['annotated']
                          if s.duration > self.duration]

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
        self.original_classes_ = sorted(labels)

        for current_file in getattr(protocol, subset)():
            uri = get_unique_identifier(current_file)
            self.data_[uri]['y'] = self.initialize_y(current_file)

    @property
    def specifications(self):
        return {
            'task': None,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.original_classes_},
        }

    def _samples(self):
        if self.exhaustive:
            return self._sliding_samples()
        else:
            return self._random_samples()

    def _random_samples(self):
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
            subsegment = next(random_subsegment(segment, self.duration))

            yield {
                'X': self.feature_extraction.crop(current_file,
                                                  subsegment, mode='center',
                                                  fixed=self.duration),
                'y': self.crop_y(datum['y'], subsegment),
            }

    def _sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        sliding_segments = SlidingSegments(duration=self.duration,
                                           step=self.duration,
                                           source='annotated')

        while True:

            np.random.shuffle(uris)

            # loop on all files
            for uri in uris:

                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum['current_file'])

                # compute features for the whole file
                features = self.feature_extraction(current_file)

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                annotated = Timeline(
                    [Segment(s.start + np.random.random() * self.duration,
                             s.end) for s in get_annotated(current_file)])
                current_file['annotated'] = annotated

                if self.shuffle:
                    samples = []

                for sequence in sliding_segments.from_file(current_file):

                    X = features.crop(sequence, mode='center',
                                      fixed=self.duration)
                    y = self.crop_y(datum['y'], subsegment)
                    sample = {'X': X, 'y': y}

                    if self.shuffle:
                        samples.append(sample)
                    else:
                        yield sample

                if self.shuffle:
                    np.random.shuffle(samples)
                    for sample in samples:
                        yield sample

    @property
    def signature(self):
        return {'X': {'@': (None, np.stack)},
                'y': {'@': (None, np.stack)}}

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * 24 * 60 * 60
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    def __call__(self):
        """(Parallelized) batch generator"""

        # number of batches needed to complete an epoch
        batches_per_epoch = self.batches_per_epoch

        generators = []

        if self.parallel:
            for _ in range(self.parallel):

                # batchify sampler and make sure at least
                # `batches_per_epoch` batches are prefetched.
                batches = batchify(self._samples(), self.signature,
                                   batch_size=self.batch_size,
                                   prefetch=batches_per_epoch)

                # add batch generator to the list of (background) generators
                generators.append(batches)
        else:

            # batchify sampler without prefetching
            batches = batchify(self._samples(), self.signature,
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
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, duration=3.2, batch_size=32, per_epoch=1,
                 parallel=1):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel

    def get_batch_generator(self, feature_extraction, protocol, subset='train'):
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        return LabelingTaskGenerator(
            feature_extraction, protocol, subset=subset,
            duration=self.duration, per_epoch=self.per_epoch,
            batch_size=self.batch_size, parallel=self.parallel)

    @property
    def weight(self):
        """Class/task weights

        Returns
        -------
        weight : None or `torch.Tensor`
        """
        return None

    def on_train_start(self):
        """Set loss function (with support for class weights)

        loss_func_ = Function f(input, target, weight=None) -> loss value
        """

        self.task_type_ = self.model_.specifications['task']

        if self.task_type_ == TASK_CLASSIFICATION:
            self.n_classes_ = len(self.model_.specifications['y']['classes'])
            self.loss_func_ = F.nll_loss

        if self.task_type_ == TASK_MULTI_LABEL_CLASSIFICATION:
            self.loss_func_ = F.binary_cross_entropy

        if self.task_type_ == TASK_REGRESSION:
            def mse_loss(input, target, weight=None):
                return F.mse_loss(input, target)
            self.loss_func_ = mse_loss

    def batch_loss(self, batch):

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        fX = self.model_(X)

        if self.task_type_ == TASK_CLASSIFICATION:
            fX = fX.view((-1, self.n_classes_))

            target = torch.tensor(
                batch['y'],
                dtype=torch.int64,
                device=self.device_).contiguous().view((-1, ))

        elif self.task_type_ in [TASK_MULTI_LABEL_CLASSIFICATION,
                                 TASK_REGRESSION]:

            target = torch.tensor(
                batch['y'],
                dtype=torch.float32,
                device=self.device_)

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)

        return self.loss_func_(fX, target, weight=weight)
