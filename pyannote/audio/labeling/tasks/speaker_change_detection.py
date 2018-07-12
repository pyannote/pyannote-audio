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
import scipy.signal


class SpeakerChangeDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speaker change detection

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    collar : float, optional
        Duration of "change" collar, in seconds. Default to 100ms (0.1).
    window : {'plateau', 'triangle'}, optional
        Defaults to 'plateau'.
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

    def __init__(self, precomputed, collar=0.100, window='plateau', **kwargs):

        super(SpeakerChangeDetectionGenerator, self).__init__(
            precomputed, exhaustive=True, **kwargs)

        self.collar = collar
        self.window = window
        if window not in {'plateau', 'triangle'}:
            msg = "'window' must be one of {'plateau', 'triangle'}."
            raise ValueError(msg)

        # convert duration to number of samples
        M = self.precomputed.sliding_window.durationToSamples(self.collar)

        # triangular window
        self.window_ = scipy.signal.triang(M)[:, np.newaxis]

    def postprocess_y(self, Y):
        """Generate labels for speaker change detection

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by `pyannote.audio.util.to_numpy`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.audio.util.to_numpy`
        """

        # True = change. False = no change
        y = np.sum(np.abs(np.diff(Y, axis=0)), axis=1, keepdims=True)
        y = np.vstack(([[0]], y > 0))

        # mark change points neighborhood as positive
        y = np.minimum(1, scipy.signal.convolve(y, self.window_, mode='same'))

        if self.window == 'plateau':
            # HACK for some reason, y rarely equals zero
            return 1 * (y > 1e-10)
        elif self.window == 'triangle':
            return y


class SpeakerChangeDetection(LabelingTask):
    """Train speaker change detection

    Parameters
    ----------
    collar : float, optional
        Duration of "change" collar, in seconds. Default to 100ms (0.1).
    window : {'plateau', 'triangle'}, optional
        Defaults to 'plateau'.
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

    def __init__(self, collar=0.100, window='plateau', **kwargs):
        super(SpeakerChangeDetection, self).__init__(**kwargs)
        self.collar = collar
        self.window = window
        if window not in {'plateau', 'triangle'}:
            msg = "'window' must be one of {'plateau', 'triangle'}."
            raise ValueError(msg)

    def get_batch_generator(self, precomputed):
        return SpeakerChangeDetectionGenerator(
            precomputed, collar=self.collar, window=self.window,
            duration=self.duration, batch_size=self.batch_size,
            per_epoch=self.per_epoch, parallel=self.parallel)

    @property
    def n_classes(self):
        if self.window == 'plateau':
            return 2
        elif self.window == 'triangle':
            return 1
