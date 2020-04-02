#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

from typing import Optional
from typing import Text

import torch
import torch.nn.functional as F

import numpy as np
import scipy.signal

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.database.protocol.protocol import Protocol

from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.audio.features import RawAudio
from pyannote.audio.features.wrapper import Wrapper, Wrappable

from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.train.trainer import Trainer
from pyannote.audio.train.generator import BatchGenerator

from pyannote.audio.train.task import Task, TaskType, TaskOutput

from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.model import Alignment

SECONDS_IN_A_DAY = 24 * 60 * 60


class LabelingTaskGenerator(BatchGenerator):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    feature_extraction : Wrappable
        Describes how features should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
    protocol : Protocol
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset.
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    mask : str, optional
        When provided, current_file[mask] is used by the loss function to weigh
        samples.
    """

    def __init__(self,
                 feature_extraction: Wrappable,
                 protocol: Protocol,
                 subset: Text = 'train',
                 resolution: Optional[Resolution] = None,
                 alignment: Optional[Alignment] = None,
                 duration: float = 2.0,
                 batch_size: int = 32,
                 per_epoch: float = None,
                 exhaustive: bool = False,
                 step: float = 0.1,
                 mask: Text = None):

        self.feature_extraction = Wrapper(feature_extraction)
        self.duration = duration
        self.exhaustive = exhaustive
        self.step = step
        self.mask = mask

        self.resolution_ = resolution

        if alignment is None:
            alignment = 'center'
        self.alignment = alignment

        self.batch_size = batch_size

        # load metadata and estimate total duration of training data
        total_duration = self._load_metadata(protocol, subset=subset)

        #
        if per_epoch is None:

            # 1 epoch = covering the whole training set once
            #
            per_epoch = total_duration / SECONDS_IN_A_DAY

            # when exhaustive is False, this is not completely correct.
            # in practice, it will randomly sample audio chunk until their
            # overall duration reaches the duration of the training set.
            # but nothing guarantees that every single part of the training set
            # has been seen exactly once: it might be more than once, it might
            # be less than once. on average, however, after a certain amount of
            # epoch, this will be correct

            # when exhaustive is True, however, we can actually make sure every
            # single part of the training set has been seen. we just have to
            # make sur we account for the step used by the exhaustive sliding
            # window
            if self.exhaustive:
                per_epoch *= np.ceil(1 / self.step)

        self.per_epoch = per_epoch

    # TODO. use cached property (Python 3.8 only)
    # https://docs.python.org/fr/3/library/functools.html#functools.cached_property
    @property
    def resolution(self):

        if self.resolution_ in [None, RESOLUTION_FRAME]:
            return self.feature_extraction.sliding_window

        if self.resolution_ == RESOLUTION_CHUNK:
            return self.SlidingWindow(duration=self.duration,
                                      step=self.step * self.duration)

        return self.resolution_

    def postprocess_y(self, Y: np.ndarray) -> np.ndarray:
        """This function does nothing but return its input.
        It should be overriden by subclasses.

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray

        Returns
        -------
        postprocessed :

        """
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
                                self.resolution,
                                labels=self.segment_labels_,
                                mode='center')

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
        cropped_y : (n_samples, dim) `np.ndarray`
            y for specified `segment`
        """

        return y.crop(segment,
                      mode=self.alignment,
                      fixed=self.duration)

    def _load_metadata(self, protocol, subset='train') -> float:
        """Load training set metadata

        This function is called once at instantiation time, returns the total
        training set duration, and populates the following attributes:

        Attributes
        ----------
        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file labels in protocol

        Returns
        -------
        duration : float
            Total duration of annotated segments, in seconds.
        """

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # ensure annotation/annotated are cropped to actual file duration
            support = Segment(start=0, end=current_file['duration'])
            current_file['annotated'] = get_annotated(current_file).crop(
                support, mode='intersection')
            current_file['annotation'] = current_file['annotation'].crop(
                support, mode='intersection')

            # keep track of unique segment labels
            segment_labels.update(current_file['annotation'].labels())

            # keep track of unique file labels
            for key, value in current_file.items():
                if isinstance(value, (Annotation, Timeline, SlidingWindowFeature)):
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(value)

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

        self.file_labels_ = {k: sorted(file_labels[k]) for k in file_labels}
        self.segment_labels_ = sorted(segment_labels)

        for uri in list(self.data_):
            current_file = self.data_[uri]['current_file']
            y = self.initialize_y(current_file)
            self.data_[uri]['y'] = y
            if self.mask is not None:
                mask = current_file[self.mask]
                current_file[self.mask] = mask.align(y)

        return sum(datum['duration'] for datum in self.data_.values())

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                         output=TaskOutput.SEQUENCE),
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.segment_labels_},
        }

        return specs

    def samples(self):
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

            X = self.feature_extraction.crop(current_file,
                                             subsegment,
                                             mode='center',
                                             fixed=self.duration)

            y = self.crop_y(datum['y'],
                            subsegment)
            sample = {'X': X, 'y': y}

            if self.mask is not None:
                mask = self.crop_y(current_file[self.mask],
                                   subsegment)
                sample['mask'] = mask

            for key, classes in self.file_labels_.items():
                sample[key] = classes.index(current_file[key])

            yield sample

    def _sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)
        sliding_segments = SlidingWindow(duration=self.duration,
                                         step=self.step * self.duration)

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
                annotated = Timeline()
                for segment in get_annotated(current_file):
                    shifted_segment = Segment(
                        segment.start + np.random.random() * self.duration,
                        segment.end)
                    if shifted_segment:
                        annotated.add(shifted_segment)

                samples = []
                for sequence in sliding_segments(annotated):

                    X = features.crop(sequence, mode='center',
                                      fixed=self.duration)
                    y = self.crop_y(datum['y'], sequence)
                    sample = {'X': X, 'y': y}

                    if self.mask is not None:

                        # extract mask for current sub-segment
                        mask = current_file[self.mask].crop(sequence,
                                                            mode='center',
                                                            fixed=self.duration)

                        # it might happen that "mask" and "y" use different
                        # sliding windows. therefore, we simply resample "mask"
                        # to match "y"
                        if len(mask) != len(y):
                            mask = scipy.signal.resample(mask, len(y), axis=0)
                        sample['mask'] = mask

                    for key, classes in self.file_labels_.items():
                        sample[key] = classes.index(current_file[key])

                    samples.append(sample)

                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * SECONDS_IN_A_DAY
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


class LabelingTask(Trainer):
    """Base class for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    """

    def __init__(self, duration: float = 2.0,
                       batch_size: int = 32,
                       per_epoch: float = None,
                       exhaustive: bool = False,
                       step: float = 0.1):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.exhaustive = exhaustive
        self.step = step

    def get_batch_generator(self, feature_extraction: Wrappable,
                                  protocol: Protocol,
                                  subset: Text = 'train',
                                  resolution: Optional[Resolution] = None,
                                  alignment: Optional[Alignment] = None) -> LabelingTaskGenerator:
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : Wrappable
            Describes how features should be obtained.
            See pyannote.audio.features.wrapper.Wrapper documentation for details.
        protocol : Protocol
        subset : {'train', 'development'}, optional
            Defaults to 'train'.
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """

        return LabelingTaskGenerator(feature_extraction,
                                     protocol,
                                     subset=subset,
                                     resolution=resolution,
                                     alignment=alignment,
                                     duration=self.duration,
                                     per_epoch=self.per_epoch,
                                     batch_size=self.batch_size,
                                     exhaustive=self.exhaustive,
                                     step=self.step)

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

        self.task_ = self.model_.task

        if self.task_.is_multiclass_classification:

            self.n_classes_ = len(self.model_.classes)

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.nll_loss(input, target, weight=weight,
                                      reduction='mean')
                else:
                    return torch.mean(
                        mask * F.nll_loss(input, target,
                                          weight=weight,
                                          reduction='none'))

        if self.task_.is_multilabel_classification:

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.binary_cross_entropy(input, target, weight=weight,
                                                  reduction='mean')
                else:
                    return torch.mean(
                        mask * F.binary_cross_entropy(input, target,
                                                      weight=weight,
                                                      reduction='none'))

        if self.task_.is_regression:

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.mse_loss(input, target,
                                      reduction='mean')
                else:
                    return torch.mean(
                        mask * F.mse_loss(input, target,
                                          reduction='none'))

        self.loss_func_ = loss_func

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
            ['mask'] (`numpy.ndarray`, optional)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        fX = self.model_(X)

        mask = None
        if self.task_.is_multiclass_classification:

            fX = fX.view((-1, self.n_classes_))

            target = torch.tensor(
                batch['y'],
                dtype=torch.int64,
                device=self.device_).contiguous().view((-1, ))

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_).contiguous().view((-1, ))


        elif self.task_.is_multilabel_classification or \
             self.task_.is_regression:

            target = torch.tensor(
                batch['y'],
                dtype=torch.float32,
                device=self.device_)

            if 'mask' in batch:
                mask = torch.tensor(
                    batch['mask'],
                    dtype=torch.float32,
                    device=self.device_)

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)

        return {
            'loss': self.loss_func_(fX, target,
                                    weight=weight,
                                    mask=mask),
        }
