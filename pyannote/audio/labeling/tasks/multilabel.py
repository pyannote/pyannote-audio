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
import re
import warnings
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_LABEL_CLASSIFICATION
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.audio.features import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.core import Segment
from pyannote.core import SlidingWindowFeature


class MultilabelGenerator(LabelingTaskGenerator):
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
    >>> batches =  MultilabelGenerator(precomputed)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerRole.JSALT')

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, n_tasks) numpy array
    >>>     pass
    """

    def __init__(self, feature_extraction, overlap=False, speech=False, **kwargs):

        super(MultilabelGenerator, self).__init__(
            feature_extraction, **kwargs)
        self.overlap = overlap
        self.speech = speech

    def postprocess_y(self, Y):
        # number of speakers for each frame
        speaker_count = np.sum(Y, axis=1, keepdims=True)

        # mark speech regions as such
        speech = np.int64(speaker_count > 0)

        # Last column is speech
        # For now, it's activated only where UU* speakers
        # are speaking
        if self.speech:
            Y = Y[:, :-1]

        # We add overlap first (columns are alphabetically sorted by their CLASS id)
        if self.overlap:
            # mark overlap regions as such
            overlap = np.int64(speaker_count > 1)
            Y = np.concatenate((Y, overlap), axis=1)

        # Then we can add speech
        if self.speech:
            Y = np.concatenate((Y, speech), axis=1)

        # replace NaNs by 0s
        Y = np.nan_to_num(Y)
        return Y


class Multilabel(LabelingTask):
    """Train a 4-labels classifier

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
    >>> task = Multilabel()

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(precomputed.dimension, task.n_classes)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('BabyTrain.SpeakerRole.JSALT')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass

    """
<<<<<<< HEAD
    def __init__(self, protocol_name, preprocessors, weighted_loss=False, training=False, overlap=False, speech=False, **kwargs):
=======
    def __init__(self, protocol_name, preprocessors, weighted_loss=False, training=False, **kwargs):
>>>>>>> 33d31acebe96a74acd100688a593fc3aabc9057f
        super(Multilabel, self).__init__(**kwargs)
        # Need protocol to know the classes that need to be predicted
        # And thus the dimension of the target !
        self.protocol = get_protocol(protocol_name, preprocessors)
        self.weighted_loss = weighted_loss
        self.overlap = overlap
        self.speech = speech
        self.labels_ = self._update_labels()

    def _update_labels(self):
        """
        Get the actual number of labels (the roles amongst {CHI,FEM,KCHI,MAL} that are present
        in the data
        """
        labels = set()
        for current_file in self.protocol.train():
            y_labels = set(current_file["annotation"].labels())
            labels |= y_labels
<<<<<<< HEAD

        if self.overlap:
            labels.add("OVERLAP")
        if self.speech:
            labels.add("SPEECH")
=======
>>>>>>> 33d31acebe96a74acd100688a593fc3aabc9057f
        return sorted(labels)

    def get_batch_generator(self, feature_extraction):
        return MultilabelGenerator(
            feature_extraction, duration=self.duration,
            batch_size=self.batch_size, per_epoch=self.per_epoch,
            parallel=self.parallel, overlap=self.overlap, speech=self.speech)

    @property
    def task_type(self):
        return TASK_MULTI_LABEL_CLASSIFICATION

    @property
    def n_classes(self):
        return len(self.labels_)

    def _get_one_over_the_prior(self):
        nb_speakers = 4
        weights = dict([(key, 0.0) for key in self.labels[0:nb_speakers]])

        # Compute the cumulated speech duration
        for current_file in self.protocol.train():
            y = current_file["annotation"]
            for speaker in self.labels[0:nb_speakers]:
                weights[speaker] += y.label_duration(speaker)

        total_speech = sum(weights.values(), 0.0)
        for key, value in weights.items():
            if value != 0:
                weights[key] = total_speech/value

        # Finally normalize, so that the weights sum to 1
        # We remove speaker roles whose value is equal to 0
        norm1 = sum(weights.values())
        weights = {key: value/norm1 for key, value in weights.items() if value != 0}
<<<<<<< HEAD

        # We might want to compute priors on overlap and speech as well...
        if self.overlap:
            weights["OVERLAP"] = 1.0
        if self.speech:
            weights["SPEECH"] = 1.0
=======
>>>>>>> 33d31acebe96a74acd100688a593fc3aabc9057f
        return torch.tensor(np.array(list(weights.values())), dtype=torch.float32)

    @property
    def weight(self):
        if self.weighted_loss:
            return self._get_one_over_the_prior()
        return None

    @property
    def labels(self):
        return list(self.labels_)
