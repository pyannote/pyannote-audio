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

from pyannote.audio.features import FeatureExtraction
from pyannote.audio.features import RawAudio

from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.train.trainer import Trainer
from pyannote.audio.train.generator import BatchGenerator

from pyannote.audio.train.task import Task, TaskType, TaskOutput


class LabelingTaskGenerator(BatchGenerator):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
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
        Duration of sub-sequences. Defaults to 3.2s.
    step : `float`, optional
        Sub-sequences step. Defaults to `duration`.
        Only used when `exhaustive` is True.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    shuffle : bool, optional
        Shuffle exhaustive samples. Defaults to False.
    mask_dimension : `int`, optional
        When set, batches will have a "mask" key that provides a mask that has
        the same length as "y". This "mask" will be passed to the loss function
        has a way to weigh samples according to their "mask" value. The actual
        value of `mask_dimension` is used to select which dimension to use.
        This option assumes that `current_file["mask"]` contains a
        `SlidingWindowFeature` that can be used as masking. Defaults to not use
        masking.
    mask_logscale : `bool`, optional
        Set to True to indicate that mask values are log scaled. Will apply
        exponential. Defaults to False. Has not effect when `mask_dimension`
        is not set.

    """

    def __init__(self,
                 feature_extraction: FeatureExtraction,
                 protocol: Protocol,
                 subset='train',
                 resolution=None,
                 alignment=None,
                 duration=3.2,
                 step=None,
                 batch_size: int = 32,
                 per_epoch: float = None,
                 exhaustive=False,
                 shuffle=False,
                 mask_dimension=None,
                 mask_logscale=False):

        self.feature_extraction = feature_extraction

        if resolution is None:
            resolution = self.feature_extraction.sliding_window
        self.resolution = resolution

        if alignment is None:
            alignment = 'center'
        self.alignment = alignment

        self.duration = duration
        if step is None:
            step = duration
        self.step = step
        self.batch_size = batch_size

        self.exhaustive = exhaustive
        self.shuffle = shuffle

        self.mask_dimension = mask_dimension
        self.mask_logscale = mask_logscale

        total_duration = self._load_metadata(protocol, subset=subset)
        if per_epoch is None:
            per_epoch = total_duration / (24 * 60 * 60)
        self.per_epoch = per_epoch


    def postprocess_y(self, Y):
        """This function does nothing but return its input.
        It should be overriden by subclasses.

        Parameters
        ----------
        Y :

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

        return y.crop(segment, mode=self.alignment,
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

        for current_file in getattr(protocol, subset)():
            uri = get_unique_identifier(current_file)
            self.data_[uri]['y'] = self.initialize_y(current_file)

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
                                             subsegment, mode='center',
                                             fixed=self.duration)

            y = self.crop_y(datum['y'], subsegment)
            sample = {'X': X, 'y': y}

            if self.mask_dimension is not None:

                # extract mask for current sub-segment
                mask = current_file['mask'].crop(subsegment,
                                                 mode='center',
                                                 fixed=self.duration)

                # use requested dimension (e.g. non-overlap scores)
                mask = mask[:, self.mask_dimension]
                if self.mask_logscale:
                    mask = np.exp(mask)

                # it might happen that "mask" and "y" use different sliding
                # windows. therefore, we simply resample "mask" to match "y"
                if len(mask) != len(y):
                    mask = scipy.signal.resample(mask, len(y), axis=0)

                sample['mask'] = mask

            for key, classes in self.file_labels_.items():
                sample[key] = classes.index(current_file[key])

            yield sample

    def _sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)
        sliding_segments = SlidingWindow(duration=self.duration,
                                         step=self.step)

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

                if self.shuffle:
                    samples = []

                for sequence in sliding_segments(annotated):

                    X = features.crop(sequence, mode='center',
                                      fixed=self.duration)
                    y = self.crop_y(datum['y'], sequence)
                    sample = {'X': X, 'y': y}

                    if self.mask_dimension is not None:

                        # extract mask for current sub-segment
                        mask = current_file['mask'].crop(sequence,
                                                         mode='center',
                                                         fixed=self.duration)

                        # use requested dimension (e.g. non-overlap scores)
                        mask = mask[:, self.mask_dimension]
                        if self.mask_logscale:
                            mask = np.exp(mask)

                        # it might happen that "mask" and "y" use different
                        # sliding windows. therefore, we simply resample "mask"
                        # to match "y"
                        if len(mask) != len(y):
                            mask = scipy.signal.resample(mask, len(y), axis=0)
                        sample['mask'] = mask

                    for key, classes in self.file_labels_.items():
                        sample[key] = classes.index(current_file[key])

                    if self.shuffle:
                        samples.append(sample)
                    else:
                        yield sample

                if self.shuffle:
                    np.random.shuffle(samples)
                    for sample in samples:
                        yield sample

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * 24 * 60 * 60
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


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
    """

    def __init__(self, duration=3.2, batch_size=32, per_epoch=1):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            resolution=None, alignment=None):
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`
        protocol : `pyannote.database.Protocol`
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
        return LabelingTaskGenerator(
            feature_extraction,
            protocol,
            subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            step=self.step,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size)

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
