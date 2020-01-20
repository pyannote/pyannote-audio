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
from .base_labeling import BaseLabeling
from pyannote.database import get_annotated
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure
from pyannote.metrics.segmentation import SegmentationPurityCoverageFMeasure

from pyannote.audio.features import Pretrained
from pyannote.audio.pipeline.speaker_change_detection \
    import SpeakerChangeDetection as SpeakerChangeDetectionPipeline


def validate_helper_func(current_file, pipeline=None, metric=None):
    reference = current_file['annotation']
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file)
    return metric(reference, hypothesis, uem=uem)


class SpeakerChangeDetection(BaseLabeling):

    Pipeline = SpeakerChangeDetectionPipeline

    def validation_criterion(self, protocol, purity=0.9, **kwargs):
        return f'purity={100*purity:.0f}%'

    def validate_epoch(self, epoch,
                             validation_data,
                             device=None,
                             batch_size=32,
                             purity=0.9,
                             diarization=False,
                             n_jobs=1,
                             duration=None,
                             step=0.25,
                             **kwargs):

        # compute (and store) SCD scores
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device)

        for current_file in validation_data:
            current_file['scd_scores'] = pretrained(current_file)

        # pipeline
        pipeline = self.Pipeline(purity=purity)

        # dichotomic search to find alpha that maximizes coverage
        # while having at least `self.purity`

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_coverage = 0.

        for _ in range(10):

            current_alpha = .5 * (lower_alpha + upper_alpha)
            pipeline.instantiate({'alpha': current_alpha,
                                  'min_duration': 0.100})

            if diarization:
                metric = DiarizationPurityCoverageFMeasure(parallel=True)
            else:
                metric = SegmentationPurityCoverageFMeasure(parallel=True)

            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               metric=metric)

            if n_jobs > 1:
                _ = self.pool_.map(validate, validation_data)
            else:
                for file in validation_data:
                    _ = validate(file)

            _purity, _coverage, _ = metric.compute_metrics()
            # TODO: normalize coverage with what one could achieve if
            # we were to put all reference speech turns in its own cluster

            if _purity < purity:
                upper_alpha = current_alpha
            else:
                lower_alpha = current_alpha
                if _coverage > best_coverage:
                    best_coverage = _coverage
                    best_alpha = current_alpha

        return {'metric': f'coverage@{purity:.2f}purity',
                'minimize': False,
                'value': best_coverage if best_coverage \
                         else _purity - purity,
                'pipeline': pipeline.instantiate({'alpha': best_alpha,
                                                  'min_duration': 0.100})}
