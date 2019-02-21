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

"""Speaker change detection"""

import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_REGRESSION
from .base import TASK_CLASSIFICATION
import scipy.signal


class SpeakerChangeDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speaker change detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    variant : {'boundary', 'multiple', 'triangle'}, optional
        Defines how change point groundtruth labels are built.
        'boundary' (defaults) means time steps in the neighborhood of any
        speech turn boundary are marked as change point. 'multiple' means only
        time steps whose neighborhood contains (at least) two speakers are
        marked as change point. 'triangle' is the same a 'boundary' except
        labels are not binary but in the shape of a triangle centered on speech
        turn boundaries.
    collar : float, optional
        Duration of neighborhood, in seconds. Default to 100ms (0.1).
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

    Usage
    -----
    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/mfcc')

    # instantiate batch generator
    >>> batches = SpeakerChangeDetectionGenerator(precomputed)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, 1) numpy array
    >>>     pass
    """

    def __init__(self, feature_extraction, collar=0.100, variant='boundary',
                 **kwargs):

        super(SpeakerChangeDetectionGenerator, self).__init__(
            feature_extraction, **kwargs)

        self.collar = collar
        self.variant = variant
        if variant not in {'boundary', 'multiple', 'triangle'}:
            msg = "'variant' must be one of {boundary, multiple, triangle}."
            raise ValueError(msg)

        # number of samples in collar
        self.collar_ = \
            self.feature_extraction.sliding_window.durationToSamples(collar)
        if variant in {'multiple'}:
            self.collar_ *= 2

        # window
        if variant in {'boundary', 'triangle'}:
            self.window_ = scipy.signal.triang(self.collar_)[:, np.newaxis]

    def postprocess_y(self, Y):
        """Generate labels for speaker change detection

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        # replace NaNs by 0s
        Y = np.nan_to_num(Y)

        if self.variant in {'boundary', 'triangle'}:

            # True = change. False = no change
            y = np.sum(np.abs(np.diff(Y, axis=0)), axis=1, keepdims=True)
            y = np.vstack(([[0]], y > 0))

            # mark change points neighborhood as positive
            y = np.minimum(1, scipy.signal.convolve(y, self.window_, mode='same'))

            # HACK for some reason, y rarely equals zero
            if self.variant == 'boundary':
                y = 1 * (y > 1e-10)

        elif self.variant in {'multiple'}:

            n_samples, n_speakers = Y.shape

            # append (half collar) empty samples at the beginning/end
            expanded_Y = np.vstack([
                np.zeros(((self.collar_ + 1) // 2 , n_speakers), dtype=Y.dtype),
                Y,
                np.zeros(((self.collar_ + 1) // 2 , n_speakers), dtype=Y.dtype)])

            # stride trick. data[i] is now a sliding window of collar length
            # centered at time step i.
            data = np.lib.stride_tricks.as_strided(expanded_Y,
                shape=(n_samples, n_speakers, self.collar_),
                strides=(Y.strides[0], Y.strides[1], Y.strides[0]))

            # y[i] = 1 if more than one speaker are speaking in the
            # corresponding window. 0 otherwise
            y = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
            y = y.reshape(-1, 1)

        return y


class SpeakerChangeDetection(LabelingTask):
    """Train speaker change detection

    Parameters
    ----------
    variant : {'boundary', 'multiple', 'triangle'}, optional
        Defines how change point groundtruth labels are built.
        'boundary' (defaults) means time steps in the neighborhood of any
        speech turn boundary are marked as change point. 'multiple' means only
        time steps whose neighborhood contains (at least) two speakers are
        marked as change point. 'triangle' is the same a 'boundary' except
        labels are not binary but in the shape of a triangle centered on speech
        turn boundaries.
    collar : float, optional
        Duration of neighborhood, in seconds. Default to 100ms (0.1).
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

    Usage
    -----
    >>> task = SpeakerChangeDetection()

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(precomputed.dimension, task.n_classes)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass

    """

    def __init__(self, collar=0.100, variant='boundary', **kwargs):
        super(SpeakerChangeDetection, self).__init__(**kwargs)
        self.collar = collar
        self.variant = variant
        if variant not in {'boundary', 'multiple', 'triangle'}:
            msg = "'variant' must be one of {boundary, multiple, triangle}."
            raise ValueError(msg)


    def get_batch_generator(self, precomputed):
        return SpeakerChangeDetectionGenerator(
            precomputed, collar=self.collar, variant=self.variant,
            duration=self.duration, batch_size=self.batch_size,
            per_epoch=self.per_epoch, parallel=self.parallel)

    @property
    def n_classes(self):
        if self.variant in {'boundary', 'multiple'}:
            return 2

        elif self.variant in {'triangle'}:
            return 1

    @property
    def task_type(self):
        if self.variant in {'triangle'}:
            return TASK_REGRESSION
        return TASK_CLASSIFICATION
