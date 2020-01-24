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

from functools import partial
from .base_labeling import BaseLabeling
from pyannote.database import get_annotated
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionPrecision

from pyannote.audio.features import Pretrained
from pyannote.audio.pipeline.overlap_detection \
    import OverlapDetection as OverlapDetectionPipeline
from pyannote.core import Timeline


def validate_helper_func(current_file, pipeline=None,
                         precision=None, recall=None):
    reference = current_file['annotation']
    uem = get_annotated(current_file)
    hypothesis = pipeline(current_file)
    p = precision(reference, hypothesis, uem=uem)
    r = recall(reference, hypothesis, uem=uem)
    return p, r


class OverlapDetection(BaseLabeling):

    Pipeline = OverlapDetectionPipeline

    def validate_init(self, protocol,
                            subset='development'):

        validation_data = super().validate_init(protocol, subset=subset)
        for current_file in validation_data:

            uri = current_file['uri']

            # build overlap reference
            overlap = Timeline(uri=uri)
            turns = current_file['annotation']
            for track1, track2 in turns.co_iter(turns):
                if track1 == track2:
                    continue
                overlap.add(track1[0] & track2[0])
            current_file['annotation'] = overlap.support().to_annotation()
            # TODO. make 'annotated' focus on speech regions only
        return validation_data

    def validation_criterion(self, protocol, precision=0.9, **kwargs):
        return f'precision={100*precision:.0f}%'

    def validate_epoch(self,
                       epoch,
                       validation_data=None,
                       device=None,
                       batch_size=32,
                       precision=0.9,
                       n_jobs=1,
                       duration=None,
                       step=0.25,
                       **kwargs):

        # compute (and store) overlap scores
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device)

        for current_file in validation_data:
            current_file['ovl_scores'] = pretrained(current_file)

        # pipeline
        pipeline = self.Pipeline(precision=precision)

        # dichotomic search to find threshold that maximizes recall
        # while having at least `target_precision`

        lower_alpha = 0.
        upper_alpha = 1.
        best_alpha = .5 * (lower_alpha + upper_alpha)
        best_recall = 0.

        for _ in range(10):

            current_alpha = .5 * (lower_alpha + upper_alpha)
            pipeline.instantiate({'onset': current_alpha,
                                  'offset': current_alpha,
                                  'min_duration_on': 0.100,
                                  'min_duration_off': 0.100,
                                  'pad_onset': 0.,
                                  'pad_offset': 0.})

            _precision = DetectionPrecision(parallel=True)
            _recall = DetectionRecall(parallel=True)

            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               precision=_precision,
                               recall=_recall)

            if n_jobs > 1:
                _ = self.pool_.map(validate, validation_data)
            else:
                for file in validation_data:
                    _ = validate(file)

            _precision = abs(_precision)
            _recall = abs(_recall)

            if not _recall:
                # lower the threshold until we at least return something...
                upper_alpha = current_alpha
                best_alpha = current_alpha
                _precision = 0.

            elif _precision < precision:
                # increase the threshold while precision is not good enough
                lower_alpha = current_alpha

            else:
                # lower the threshold if we return something and
                # precision is good enough
                upper_alpha = current_alpha
                if _recall > best_recall:
                    best_recall = _recall
                    best_alpha = current_alpha

        return {'metric': f'recall@{precision:.2f}precision',
                'minimize': False,
                'value': best_recall if best_recall \
                         else _precision - precision,
                'pipeline': pipeline.instantiate({'onset': best_alpha,
                                                  'offset': best_alpha,
                                                  'min_duration_on': 0.100,
                                                  'min_duration_off': 0.100,
                                                  'pad_onset': 0.,
                                                  'pad_offset': 0.})}
