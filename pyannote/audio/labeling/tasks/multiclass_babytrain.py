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
4-way classification :
KCHI    (the key child wearing the device)
CHI     (other children)
FEM     (female speech)
MAL     (male speech)
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

    def __init__(self, feature_extraction, **kwargs):

        super(MulticlassBabyTrainGenerator, self).__init__(
            feature_extraction, **kwargs)
        #self.overlap = overlap
        self.overlap = True

    def postprocess_y(self, Y):
        """Add speech to Y

        Parameters
        ----------
        Y : (n_samples, n_speaker_classes) numpy.ndarray
            Discretized annotation returned by `pyannote.core.utils_rttm.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, n_speakers_classes+0 or +1 ) numpy.ndarray if self.speech == True

        See also
        --------
        `pyannote.core.utils_rttm.numpy.one_hot_encoding`
        """
        # replace NaNs by 0s
        Y = np.nan_to_num(Y)
        return Y


class MulticlassBabyTrain(LabelingTask):
    """Train a 6-class classifier

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

    def __init__(self, weighted_loss=False, **kwargs):
        super(MulticlassBabyTrain, self).__init__(**kwargs)
        self.weighted_loss = float(weighted_loss)

    def get_batch_generator(self, feature_extraction):
        return MulticlassBabyTrainGenerator(
            feature_extraction, duration=self.duration,
            batch_size=self.batch_size, per_epoch=self.per_epoch,
            parallel=self.parallel)

    @property
    def task_type(self):
        return TASK_MULTI_LABEL_CLASSIFICATION

    @property
    def n_classes(self):
        return 4 # KCHI, CHI, FEM, MAL

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

    @property
    def labels(self):
        return ["CHI", "FEM", "KCHI", "MAL"]
