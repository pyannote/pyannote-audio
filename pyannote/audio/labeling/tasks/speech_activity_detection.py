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
from .base import TASK_CLASSIFICATION


class SpeechActivityDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speech activity detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    overlap : bool, optional
        Switch to 3 classes "non-speech vs. one speaker vs. 2+ speakers".
        Defaults to 2 classes "non-speech vs. speech".
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

    def __init__(self, feature_extraction, protocol, subset='train',
                 overlap=False, **kwargs):

        self.overlap = overlap
        super().__init__(
            feature_extraction, protocol, subset=subset, **kwargs)

    def postprocess_y(self, Y):
        """Generate labels for speech activity detection

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

        # number of speakers for each frame
        speaker_count = np.sum(Y, axis=1, keepdims=True)

        # mark speech regions as such
        speech = np.int64(speaker_count > 0)
        if self.overlap:
            # mark overlap regions as such
            overlap = np.int64(speaker_count > 1)
            return speech + overlap

        return speech

    @property
    def specifications(self):
        classes = ['non_speech', 'speech']
        if self.overlap:
            classes.append('overlap')
        return {
            'task': TASK_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': classes},
        }


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
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, overlap=False, **kwargs):
        super().__init__(**kwargs)
        self.overlap = overlap

    def get_batch_generator(self, feature_extraction, protocol, subset='train'):
        return SpeechActivityDetectionGenerator(
            feature_extraction,
            protocol, subset=subset,
            overlap=self.overlap,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel)
