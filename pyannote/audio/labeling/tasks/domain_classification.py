#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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

"""Domain classification"""

import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_CLASS_CLASSIFICATION


class DomainClassificationGenerator(LabelingTaskGenerator):
    """Batch generator for training domain classification

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
    domain : `str`, optional
        Key to use as domain. Defaults to 'domain'.
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
                 frame_info=None, frame_crop=None, domain='domain',
                 **kwargs):

        self.domain = domain
        super().__init__(
            feature_extraction, protocol, subset=subset,
            frame_info=frame_info, frame_crop=frame_crop,
            **kwargs)

    def initialize_y(self, current_file):
        return self.file_labels_[self.domain].index(current_file[self.domain])

    def crop_y(self, y, segment):
        return y

    @property
    def specifications(self):
        return {
            'task': TASK_MULTI_CLASS_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.file_labels_[self.domain]},
        }


class DomainClassification(LabelingTask):
    """Train domain classification

    Parameters
    ----------
    domain : `str`, optional
        Key to use as domain. Defaults to 'domain'.
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

    def __init__(self, domain='domain', **kwargs):
        super().__init__(**kwargs)
        self.domain = domain

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):
        """
        frame_info : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        frame_crop : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.

        """
        return DomainClassificationGenerator(
            feature_extraction,
            protocol, subset=subset,
            frame_info=frame_info,
            frame_crop=frame_crop,
            domain=self.domain,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel)
