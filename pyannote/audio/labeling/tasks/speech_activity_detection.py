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

"""Speech activity detection"""

import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator


class SpeechActivityDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speech activity detection

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    overlap : bool, optional
        Switch to 3 classes "non-speech vs. one speaker vs. 2+ speakers".
        Defaults to 2 classes "non-speech vs. speech".
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


    Usage
    -----
    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/mfcc')

    # instantiate batch generator
    >>> batches = SpeechActivityDetectionGenerator(precomputed)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, 1) numpy array
    >>>     pass
    """

    def __init__(self, precomputed, overlap=False, **kwargs):
        super(SpeechActivityDetectionGenerator, self).__init__(
            precomputed, exhaustive=True, **kwargs)
        self.overlap = overlap

    def postprocess_y(self, Y):
        """Generate labels for speech activity detection

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

        # number of speakers for each frame
        speaker_count = np.sum(Y, axis=1, keepdims=True)

        # mark speech regions as such
        speech = np.int64(speaker_count > 0)
        if self.overlap:
            # mark overlap regions as such
            overlap = np.int64(speaker_count > 1)
            return speech + overlap

        return speech


class SpeechActivityDetection(LabelingTask):
    """Train speech activity (and overlap) detection

    Parameters
    ----------
    overlap : bool, optional
        Switch to 3 classes "non-speech vs. one speaker vs. 2+ speakers".
        Defaults to 2 classes "non-speech vs. speech".
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
        Defaults to 'rmsprop'.

    Usage
    -----
    >>> task = SpeechActivityDetection()

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

    def __init__(self, overlap=False, **kwargs):
        super(SpeechActivityDetection, self).__init__(**kwargs)
        self.overlap = overlap

    def get_batch_generator(self, precomputed):
        return SpeechActivityDetectionGenerator(
            precomputed, overlap=self.overlap, duration=self.duration,
            per_epoch=self.per_epoch, batch_size=self.batch_size,
            parallel=self.parallel)

    @property
    def n_classes(self):
        return 3 if self.overlap else 2
