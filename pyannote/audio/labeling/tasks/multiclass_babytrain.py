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
# Marvin Lavechin - marvinlavechin@gmail.com

"""BabyTrain
6-way classification :
SIL     (silence)
KCHI    (the key child wearing the device)
CHI     (other children)
FEM     (female speech)
MAL     (male speech)
OVL     (overlap)
"""

import torch
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_LABEL_CLASSIFICATION
from pyannote.database import get_protocol
protocol = get_protocol('BabyTrain.SpeakerDiarization.BB')
import scipy.signal
import sys


class MulticlassBabyTrainGenerator(LabelingTaskGenerator):
    """Batch generator for training a multi-class classifier on BabyTrain

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
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
    >>> batches =  MulticlassBabyTrainGenerator(precomputed)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerDiarization.BB')

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, n_tasks) numpy array
    >>>     pass
    """

    def __init__(self, feature_extraction, overlap=True, **kwargs):

        super(MulticlassBabyTrainGenerator, self).__init__(
            feature_extraction, **kwargs)
        self.overlap = overlap

    def postprocess_y(self, Y):
        """Add overlap to Y

        Parameters
        ----------
        Y : (n_samples, n_speaker_classes) numpy.ndarray
            Discretized annotation returned by `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, n_speakers_classes+0 or +1) numpy.ndarray extended OVL if self.overlap == True

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """
        # replace NaNs by 0s
        Y = np.nan_to_num(Y)

        # Add overlap class
        if self.overlap:
            # Number of speakers
            count = np.sum(Y, axis=1)

            # Count number of speakers
            y_overlap = count > 1

            # When there's overlap, we turn off the speaker columns..
            Y[y_overlap] = np.zeros(Y[y_overlap].shape)

            # ... and turn on the overlap column !
            Y = np.column_stack((Y, 1*y_overlap))

        # Number of speakers
        # count = np.sum(Y, axis=1)
        # y_non_speech = count == 0
        # Y = np.column_stack((Y, y_non_speech))

        return Y #np.argwhere(Y == 1)[:, -1]



class MulticlassBabyTrain(LabelingTask):
    """Train a 6-class classifier

    Parameters
    ----------
    overlap : `bool` or `float`, optional
        Use overlapping speech detection task with weight `overlap`.
        Defaults to True (= 1).
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
    >>> task = MulticlassBabyTrain()

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(precomputed.dimension, task.n_classes)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerDiarization.BB')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass

    """

    def __init__(self, overlap=False, weighted_loss=False, **kwargs):
        super(MulticlassBabyTrain, self).__init__(**kwargs)
        self.overlap = float(overlap)
        self.weighted_loss = float(weighted_loss)

    def get_batch_generator(self, feature_extraction):
        return MulticlassBabyTrainGenerator(
            feature_extraction, overlap=self.overlap > 0.,  duration=self.duration,
            batch_size=self.batch_size, per_epoch=self.per_epoch,
            parallel=self.parallel)

    @property
    def task_type(self):
        return TASK_MULTI_LABEL_CLASSIFICATION

    @property
    def n_classes(self):
        return 5 if self.overlap else 4

    def _get_one_over_the_prior(self):
        nb_speakers = 4
        weights = dict([(key, 0.0) for key in self.labels[0:nb_speakers]])

        for current_file in protocol.trn_iter():
            y = current_file["annotation"]
            for speaker in self.labels[0:nb_speakers]:
                weights[speaker] += y.label_duration(speaker)

        total_speech = sum(weights.values(), 0.0)
        weights = {key: total_speech / value for key, value in weights.items()}
        return torch.tensor(np.array(list(weights.values())), dtype=torch.float32)

    @property
    def weight(self):
        if self.weighted_loss:
            return self._get_one_over_the_prior()
        return None
        #return torch.tensor(np.array(weight) / np.sum(weight),dtype=torch.float32)

    @property
    def labels(self):
        labels = ["CHI", "FEM", "KCHI", "MAL"]
        if self.overlap:
            labels.append("OVL")
        return labels
