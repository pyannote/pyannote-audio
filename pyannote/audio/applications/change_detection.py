#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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
# Ruiqing YIN - yin@limsi.fr
# Hervé BREDIN - http://herve.niderb.fr

from functools import partial
import scipy.optimize
from .base_labeling import BaseLabeling
from pyannote.database import get_annotated

from pyannote.audio.features import Pretrained
from pyannote.audio.pipeline.speaker_change_detection import (
    SpeakerChangeDetection as SpeakerChangeDetectionPipeline,
)


def validate_helper_func(current_file, pipeline=None, metric=None):
    reference = current_file["annotation"]
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file)
    return metric(reference, hypothesis, uem=uem)


class SpeakerChangeDetection(BaseLabeling):

    Pipeline = SpeakerChangeDetectionPipeline

    def validation_criterion(self, protocol, diarization=False, **kwargs):
        if diarization:
            return "diarization_fscore"
        else:
            return "segmentation_fscore"

    def validate_epoch(
        self,
        epoch,
        validation_data,
        device=None,
        batch_size=32,
        diarization=False,
        n_jobs=1,
        duration=None,
        step=0.25,
        **kwargs
    ):

        # compute (and store) SCD scores
        pretrained = Pretrained(
            validate_dir=self.validate_dir_,
            epoch=epoch,
            duration=duration,
            step=step,
            batch_size=batch_size,
            device=device,
        )

        for current_file in validation_data:
            current_file["scores"] = pretrained(current_file)

        # pipeline
        pipeline = self.Pipeline(scores="@scores", fscore=True, diarization=diarization)

        def fun(threshold):
            pipeline.instantiate({"alpha": threshold, "min_duration": 0.100})
            metric = pipeline.get_metric(parallel=True)
            validate = partial(validate_helper_func, pipeline=pipeline, metric=metric)
            if n_jobs > 1:
                _ = self.pool_.map(validate, validation_data)
            else:
                for file in validation_data:
                    _ = validate(file)

            return 1.0 - abs(metric)

        res = scipy.optimize.minimize_scalar(
            fun, bounds=(0.0, 1.0), method="bounded", options={"maxiter": 10}
        )

        threshold = res.x.item()

        return {
            "metric": self.validation_criterion(None, diarization=diarization),
            "minimize": False,
            "value": float(1.0 - res.fun),
            "pipeline": pipeline.instantiate(
                {"alpha": threshold, "min_duration": 0.100}
            ),
        }
