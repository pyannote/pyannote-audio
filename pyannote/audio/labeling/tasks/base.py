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

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.train.trainer import Trainer
from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.core import SlidingWindowFeature
from pyannote.core import Timeline
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier
from pyannote.generators.batch import batchify
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment

from .. import TASK_MULTI_CLASS_CLASSIFICATION
from .. import TASK_MULTI_LABEL_CLASSIFICATION
from .. import TASK_REGRESSION


class LabelingTaskGenerator:
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    frame_info : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    frame_crop : {'center', 'loose', 'strict'}, optional
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
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1. Each
        generator will prefetch enough batches to cover a whole epoch. Set
        `parallel` to 0 to not use background generators.
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

    def __init__(self, feature_extraction, protocol, subset='train',
                 frame_info=None, frame_crop=None,
                 duration=3.2, step=None,
                 batch_size=32, per_epoch=1, parallel=1,
                 exhaustive=False, shuffle=False,
                 mask_dimension=None, mask_logscale=False):

        self.feature_extraction = feature_extraction

        if frame_info is None:
            frame_info = self.feature_extraction.sliding_window
        self.frame_info = frame_info

        if frame_crop is None:
            frame_crop = 'center'
        self.frame_crop = frame_crop

        self.duration = duration
        if step is None:
            step = duration
        self.step = step
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.exhaustive = exhaustive
        self.shuffle = shuffle

        self.mask_dimension = mask_dimension
        self.mask_logscale = mask_logscale

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
                                self.frame_info,
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

        return y.crop(segment, mode=self.frame_crop,
                      fixed=self.duration)

    def _load_metadata(self, protocol, subset='train'):
        """Gather the following information about the training subset:

        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file labels in protocol
        """

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # ensure annotation/annotated are cropped to actual file duration
            support = Segment(start=0, end=get_audio_duration(current_file))
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

    @property
    def signature(self):
        signature = {'X': {'@': (None, np.stack)},
                     'y': {'@': (None, np.stack)}}

        if self.mask_dimension is not None:
            signature['mask'] = {'@': (None, np.stack)}

        for key in self.file_labels_:
            signature[key] = {'@': (None, np.stack)}

        return signature

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`str`) : task name
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            'task': None,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.segment_labels_},
        }

        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs

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

        sliding_segments = SlidingSegments(duration=self.duration,
                                           step=self.step,
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
                annotated = Timeline()
                for segment in get_annotated(current_file):
                    shifted_segment = Segment(
                        segment.start + np.random.random() * self.duration,
                        segment.end)
                    if shifted_segment:
                        annotated.add(shifted_segment)
                current_file['annotated'] = annotated

                if self.shuffle:
                    samples = []

                for sequence in sliding_segments.from_file(current_file):

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


    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`
        protocol : `pyannote.database.Protocol`
        subset : {'train', 'development'}, optional
            Defaults to 'train'.
        frame_info : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        frame_crop : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        return LabelingTaskGenerator(
            feature_extraction, protocol, subset=subset,
            frame_info=frame_info, frame_crop=frame_crop,
            duration=self.duration, step=self.step, per_epoch=self.per_epoch,
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

        if self.task_type_ == TASK_MULTI_CLASS_CLASSIFICATION:

            self.n_classes_ = len(self.model_.specifications['y']['classes'])

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.nll_loss(input, target, weight=weight,
                                      reduction='mean')
                else:
                    return torch.mean(
                        mask * F.nll_loss(input, target,
                                          weight=weight,
                                          reduction='none'))

        if self.task_type_ == TASK_MULTI_LABEL_CLASSIFICATION:

            def loss_func(input, target, weight=None, mask=None):
                if mask is None:
                    return F.binary_cross_entropy(input, target, weight=weight,
                                                  reduction='mean')
                else:
                    return torch.mean(
                        mask * F.binary_cross_entropy(input, target,
                                                      weight=weight,
                                                      reduction='none'))

        if self.task_type_ == TASK_REGRESSION:

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
        if self.task_type_ == TASK_MULTI_CLASS_CLASSIFICATION:

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


        elif self.task_type_ in [TASK_MULTI_LABEL_CLASSIFICATION,
                                 TASK_REGRESSION]:

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
