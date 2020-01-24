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

"""Speaker change detection"""

import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.audio.train.task import Task, TaskType, TaskOutput
import scipy.signal


class SpeakerChangeDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speaker change detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
    collar : float, optional
        Duration of positive collar, in seconds. Default to 0.1 (i.e. frames
        less than 100ms away from the actual change are also labeled as
        change).
    regression : bool, optional
        Use triangle-shaped label sequences centered on actual changes.
        Defaults to False (i.e. rectangle-shaped label sequences).
    non_speech : bool, optional
        Keep non-speech/speaker changes (and vice-versa). Defauls to False
        (i.e. only keep speaker/speaker changes).
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    """

    def __init__(self,
                 feature_extraction,
                 protocol,
                 subset='train',
                 resolution=None,
                 alignment=None,
                 collar=0.100,
                 regression=False,
                 non_speech=False,
                 **kwargs):

        self.collar = collar
        self.regression = regression
        self.non_speech = non_speech

        # number of samples in collar
        if resolution is None:
            resolution = feature_extraction.sliding_window
        self.collar_ = resolution.durationToSamples(collar)

        # window
        self.window_ = scipy.signal.triang(self.collar_)[:, np.newaxis]

        super(SpeakerChangeDetectionGenerator, self).__init__(
            feature_extraction, protocol, subset=subset,
            resolution=resolution, alignment=alignment, **kwargs)


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
        n_samples, n_speakers = Y.shape

        # True = change. False = no change
        y = np.sum(np.abs(np.diff(Y, axis=0)), axis=1, keepdims=True)
        y = np.vstack(([[0]], y > 0))

        # mark change points neighborhood as positive
        y = np.minimum(1, scipy.signal.convolve(y, self.window_, mode='same'))

        # HACK for some reason, y rarely equals zero
        if not self.regression:
            y = 1 * (y > 1e-10)

        # at this point, all segment boundaries are marked as change
        # (including non-speech/speaker changesà

        # remove non-speech/speaker change
        if not self.non_speech:

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
            x_speakers = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
            x_speakers = x_speakers.reshape(-1, 1)

            y *= x_speakers

        return y

    @property
    def specifications(self):
        if self.regression:
            return {
                'task': Task(type=TaskType.REGRESSION,
                             output=TaskOutput.SEQUENCE),
                'X': {'dimension': self.feature_extraction.dimension},
                'y': {'classes': ['change', ]},
            }

        else:
            return {
                'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                             output=TaskOutput.SEQUENCE),
                'X': {'dimension': self.feature_extraction.dimension},
                'y': {'classes': ['non_change', 'change']},
            }


class SpeakerChangeDetection(LabelingTask):
    """Train speaker change detection

    Parameters
    ----------
    collar : float, optional
        Duration of positive collar, in seconds. Default to 0.1 (i.e. frames
        less than 100ms away from the actual change are also labeled as
        change).
    regression : bool, optional
        Use triangle-shaped label sequences centered on actual changes.
        Defaults to False (i.e. rectangle-shaped label sequences).
    non_speech : bool, optional
        Keep non-speech/speaker changes (and vice-versa). Defauls to False
        (i.e. only keep speaker/speaker changes).
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    """

    def __init__(self, collar=0.100, regression=False, non_speech=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.collar = collar
        self.regression = regression
        self.non_speech = non_speech

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            resolution=None, alignment=None):
        """
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return SpeakerChangeDetectionGenerator(
            feature_extraction,
            protocol,
            resolution=resolution,
            alignment=alignment,
            subset='train',
            collar=self.collar,
            regression=self.regression,
            non_speech=self.non_speech,
            duration=self.duration,
            batch_size=self.batch_size,
            per_epoch=self.per_epoch)
