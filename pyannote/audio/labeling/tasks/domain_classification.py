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
from pyannote.audio.train.task import Task, TaskType, TaskOutput


class DomainClassificationGenerator(LabelingTaskGenerator):
    """Batch generator for training domain classification

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    domain : `str`, optional
        Key to use as domain. Defaults to 'domain'.
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
                 domain='domain',
                 **kwargs):

        self.domain = domain
        super().__init__(feature_extraction, protocol, subset=subset, **kwargs)

    def initialize_y(self, current_file):
        return self.file_labels_[self.domain].index(current_file[self.domain])

    def crop_y(self, y, segment):
        return y

    @property
    def specifications(self):
        return {
            'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                         output=TaskOutput.VECTOR),
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
    """

    def __init__(self, domain='domain', **kwargs):
        super().__init__(**kwargs)
        self.domain = domain

    def get_batch_generator(self,
                            feature_extraction,
                            protocol,
                            subset='train',
                            **kwargs):
        """Get batch generator for domain classification

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`
            Feature extraction.
        protocol : `pyannote.database.Protocol`
        subset : {'train', 'development', 'test'}, optional
            Protocol and subset used for batch generation.

        Returns
        -------
        batch_generator : `DomainClassificationGenerator`
            Batch generator
        """
        return DomainClassificationGenerator(
            feature_extraction,
            protocol,
            subset=subset,
            domain=self.domain,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size)
