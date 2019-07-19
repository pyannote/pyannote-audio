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

from itertools import cycle

import numpy as np
import torch
from pyannote.audio.features import Precomputed
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature
from pyannote.core import Timeline
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.database import get_annotated
from pyannote.database import get_protocol

from .base import LabelingTask
from .base import LabelingTaskGenerator
from .. import TASK_MULTI_LABEL_CLASSIFICATION


def derives_label(annotation, derivation_type, meta_label, regular_labels):
    """
    Derives a label. The derivation takes as inputs :
    - An annotation from which we want to derive
    - A derivation type : union or intersection
    - A meta label : the name of the output label
    - A list of regular labels : the regular labels from which we want to derive

    Example :
        1) derives_label(annotation, 'union', 'speech', ["CHI","MAL","FEM"]
        Will compute the speech label based on the union of "CHI", "MAL" and "FEM"
        2) derives_label(annotation, 'intersection', 'overlap', ["CHI","MAL","FEM"]
        Will compute the overlapping speech based on the intersection of "CHI", "MAL" and "FEM"

    annotation:  Annotation type
        The annotation we want to derive from.
    derivation_type: string, must belong to ['union', 'intersection']
        The derivation type
    meta_label:  string
        The meta label, the name of the label returned by the derivation
    regular_labels:  list of strings
        A list of regular labels we want to derive from
    """
    if derivation_type not in ['union', 'intersection']:
        raise ValueError("Derivation type must be in ['union', 'intersection')")

    derived = Annotation()
    renaming = {k: v for k, v in zip(regular_labels, [meta_label] * len(regular_labels))}
    annotation = annotation.subset(regular_labels).rename_labels(mapping=renaming)

    if derivation_type == 'union':
        support = annotation.support()
        derived.update(support.rename_tracks())
    elif derivation_type == 'intersection':
        overlap = Timeline()
        for track1, track2 in annotation.co_iter(annotation):
            if track1 == track2:
                continue
            overlap.add(track1[0] & track2[0])
        derived = overlap.support().to_annotation(generator=cycle([meta_label]))

    return derived

class MultilabelGenerator(LabelingTaskGenerator):
    """Batch generator for training a multi-class classifier on BabyTrain

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
    frame_crop : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
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

    def __init__(self, feature_extraction, protocol, labels,
                 subset='train', frame_info=None, frame_crop=None,
                 duration=3.2, batch_size=32, per_epoch=1, parallel=1,
                 shuffle=True):

        self.labels_spec = labels
        super().__init__(feature_extraction, protocol, subset=subset,
                         frame_info=frame_info, frame_crop=frame_crop,
                         duration=duration,
                         batch_size=batch_size, per_epoch=per_epoch,
                         parallel=parallel, shuffle=shuffle)

    def initialize_y(self, current_file):
        # First, one hot encode the regular classes
        annotation = current_file['annotation'].subset(self.labels_spec['regular'])
        y, _ = one_hot_encoding(annotation,
                                get_annotated(current_file),
                                self.frame_info,
                                labels=self.labels_spec["regular"],
                                mode='center')
        y_data = y.data
        # Then, one hot encode the meta classes
        for derivation_type in ['union', 'intersection']:
            for meta_label, regular_labels in self.labels_spec[derivation_type].items():
                derived = derives_label(current_file["annotation"], derivation_type, meta_label, regular_labels)
                z, _ = one_hot_encoding(derived, get_annotated(current_file),
                                        self.frame_info,
                                        labels=[meta_label],
                                        mode='center')

                y_data = np.hstack((y_data, z.data))

        return SlidingWindowFeature(self.postprocess_y(y_data),
                                    y.sliding_window)

    @property
    def specifications(self):
        return {
            'task': TASK_MULTI_LABEL_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.labels_spec["regular"]
                             + list(self.labels_spec['union'].keys())
                             + list(self.labels_spec['intersection'].keys())},
        }


class Multilabel(LabelingTask):
    """
    Train a n-labels classifier where the labels are provided by the user, and can be of 3 types :

    - Regular labels : those are computed directly from the annotation and are kept unchanged.
    - Union meta-label : those are computed by taking the union of multiple regular labels
    - Intersection meta-label : those are computed by taking the intersection of multiple regular labels.

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
    labels:
        regular: list
            List of classes that need to be predicted.
        union:
            Dictionnary of union meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
        intersection:
            Dictionnary of intersection meta-labels whose keys are the meta-label names,
            and values are a list of regular classes
    weighted_loss: bool, optional, default to False
        Compute weights that will be send later to the model.
        For each of the regular classes : 1/prior
        For each of the union/intersection classes : 1

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
    >>> protocol = get_protocol('BabyTrain.SpeakerDiarization.All')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass
    """
    def __init__(self, protocol_name, preprocessors, labels, weighted_loss=False, **kwargs):
        super(Multilabel, self).__init__(**kwargs)

        # Need protocol to know the classes that need to be predicted
        # And thus the dimension of the target !
        self.weighted_loss = weighted_loss

        # Labels related attributes
        self.labels_spec = labels
        self.label_names = labels["regular"] \
                           + list(labels['union'].keys()) \
                           + list(labels['intersection'].keys())
        self.nb_labels = len(labels)
        self.nb_regular_labels = len(labels["regular"])

        # Protocol, so that we can loop through the training set
        # and compute the prior
        self.protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):
        return MultilabelGenerator(
            feature_extraction,
            protocol, subset=subset,
            frame_info=frame_info,
            frame_crop=frame_crop,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel,
            labels=self.labels_spec)

    def _get_one_over_the_prior(self):
        weights = {k: 0.0 for k in self.label_names[0:self.nb_regular_labels]}

        # Compute the cumulated duration
        for current_file in self.protocol.train():
            y = current_file["annotation"]
            for speaker in self.label_names[0:self.nb_regular_labels]:
                weights[speaker] += y.label_duration(speaker)

        total_speech = sum(weights.values(), 0.0)
        for key, value in weights.items():
            if value != 0:
                weights[key] = total_speech/value

        # Finally normalize, so that the weights sum to 1
        norm1 = sum(weights.values())
        regular_weights = {key: value/norm1 for key, value in weights.items()}
        meta_weights = {key: 1 for key in list(self.labels_spec["union"].keys())
                        + list(self.labels_spec["intersection"].keys())}

        weights = list(regular_weights.values()) + list(meta_weights.values())
        return torch.tensor(np.array(weights), dtype=torch.float32)

    @property
    def weight(self):
        if self.weighted_loss:
            return self._get_one_over_the_prior()
        return None
