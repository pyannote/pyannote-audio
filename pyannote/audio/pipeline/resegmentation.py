#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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

raise NotImplementedError("FIXME")

from typing import Optional
from pathlib import Path
import yaml

import torch
from functools import partial
from pyannote.core.utils.helper import get_class_by_name

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Integer
from pyannote.pipeline.parameter import LogUniform
from pyannote.pipeline.parameter import Uniform

from pyannote.core import Annotation
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

from pyannote.audio.labeling.tasks.resegmentation import (
    Resegmentation as _Resegmentation,
)
from pyannote.audio.labeling.tasks.resegmentation import (
    ResegmentationWithOverlap as _ResegmentationWithOverlap,
)


class Resegmentation(Pipeline):
    """Resegmentation pipeline

    Parameters
    ----------
    feature_extraction : `dict`, optional
        Configuration dict for feature extraction.
    architecture : `dict`, optional
        Configuration dict for network architecture.
    overlap : `boolean`, optional
        Assign overlapped speech segments. Defaults to False.
    keep_sad: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    mask : `dict`, optional
        Configuration dict for masking.
        - dimension : `int`, optional
        - log_scale : `bool`, optional
    augmentation : `bool`, optional
        Augment (self-)training data by adding noise from non-speech regions.
        Defaults to False.
    duration : `float`, optional
        Defaults to 2s.
    batch_size : `int`, optional
        Defaults to 32.
    gpu : `boolean`, optional
        Defaults to False.

    Sample configuration file
    -------------------------
    pipeline:
        name: ResegmentationPipeline
        params:
            duration: 3
            batch_size: 32
            gpu: True
            overlap: True
            keep_sad: True
            feature_extraction:
               name: Precomputed
               params:
                  root_dir: /path/to/precomputed/features
            architecture:
               name: StackedLSTM
               params:
                  rnn: LSTM
            mask:
                dimension: 0
                log_scale: True

    preprocessors:
        audio: /path/to/database.yml
        hypothesis:
           name: pyannote.database.util.RTTMLoader
           params:
              train: /path/to/input.train.rttm
              development: /path/to/input.development.rttm
              test: /path/to/input.test.rttm
        overlap:
           name: pyannote.audio.features.Precomputed
           params:
              root_dir: /path/to/precomputed/overlap_scores
        mask:
           name: pyannote.audio.features.Precomputed
           params:
              root_dir: /path/to/precomputed/overlap_scores

    """

    CONFIG_YML = "{experiment_dir}/config.yml"

    # TODO. add support for data augmentation
    def __init__(
        self,
        feature_extraction: Optional[dict] = None,
        architecture: Optional[dict] = None,
        overlap: Optional[bool] = False,
        keep_sad: Optional[bool] = False,
        mask: Optional[dict] = None,
        augmentation: Optional[bool] = False,
        duration: Optional[float] = 2.0,
        batch_size: Optional[float] = 32,
        gpu: Optional[bool] = False,
    ):

        # feature extraction
        if feature_extraction is None:
            from pyannote.audio.features import LibrosaMFCC

            self.feature_extraction_ = LibrosaMFCC(
                e=False,
                De=True,
                DDe=True,
                coefs=19,
                D=True,
                DD=True,
                duration=0.025,
                step=0.010,
                sample_rate=16000,
            )
        else:
            FeatureExtraction = get_class_by_name(
                feature_extraction["name"],
                default_module_name="pyannote.audio.features",
            )
            self.feature_extraction_ = FeatureExtraction(
                **feature_extraction.get("params", {}), augmentation=None
            )

        # network architecture
        if architecture is None:
            from pyannote.audio.models import PyanNet

            self.Architecture_ = PyanNet
            self.architecture_params_ = {"sincnet": {"skip": True}}

        else:
            self.Architecture_ = get_class_by_name(
                architecture["name"], default_module_name="pyannote.audio.models"
            )
            self.architecture_params_ = architecture.get("params", {})

        self.overlap = overlap
        self.keep_sad = keep_sad

        self.mask = mask
        if mask is None:
            self.mask_dimension_ = None
            self.mask_logscale_ = False
        else:
            self.mask_dimension_ = mask["dimension"]
            self.mask_logscale_ = mask["log_scale"]

        self.augmentation = augmentation

        self.duration = duration
        self.batch_size = batch_size
        self.gpu = gpu
        self.device_ = torch.device("cuda") if self.gpu else torch.device("cpu")

        # hyper-parameters
        self.learning_rate = LogUniform(1e-3, 1)
        self.epochs = Integer(10, 50)
        self.ensemble = Integer(1, 5)
        if self.overlap:
            self.overlap_threshold = Uniform(0, 1)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        ensemble = min(self.epochs, self.ensemble)

        if self.overlap:
            self._resegmentation = _ResegmentationWithOverlap(
                self.feature_extraction_,
                self.Architecture_,
                self.architecture_params_,
                keep_sad=self.keep_sad,
                mask_dimension=self.mask_dimension_,
                mask_logscale=self.mask_logscale_,
                overlap_threshold=self.overlap_threshold,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                ensemble=ensemble,
                device=self.device_,
                duration=self.duration,
                batch_size=self.batch_size,
            )

        else:
            self._resegmentation = _Resegmentation(
                self.feature_extraction_,
                self.Architecture_,
                self.architecture_params_,
                keep_sad=self.keep_sad,
                mask_dimension=self.mask_dimension_,
                mask_logscale=self.mask_logscale_,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                ensemble=ensemble,
                device=self.device_,
                duration=self.duration,
                batch_size=self.batch_size,
            )

    def __call__(self, current_file: dict) -> Annotation:
        """Apply resegmentation

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol. Should contain a
            'hypothesis' key providing diarization before resegmentation (and
            a 'overlap' key in case overlap handling).

        Returns
        -------
        new_hypothesis : `pyannote.core.Annotation`
            Resegmented hypothesis.
        """

        return self._resegmentation.apply(current_file)

    def get_metric(self) -> GreedyDiarizationErrorRate:
        """Return new instance of detection error rate metric"""
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
