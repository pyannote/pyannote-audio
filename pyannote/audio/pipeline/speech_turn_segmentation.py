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
# Hervé BREDIN - http://herve.niderb.fr

from typing import Optional
from typing import Union
from pathlib import Path

from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from .speaker_change_detection import SpeakerChangeDetection
from .speech_activity_detection import SpeechActivityDetection
from .speech_activity_detection import OracleSpeechActivityDetection

from pyannote.database import get_annotated
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure


class OracleSpeechTurnSegmentation(Pipeline):
    """Oracle segmentation"""

    def __call__(self, current_file: dict) -> Annotation:
        """Return groundtruth segmentation

        Parameter
        ---------
        current_file : `dict`
            Dictionary as provided by `pyannote.database`.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speech turns
        """

        return current_file['annotation'].relabel_tracks(generator='string')


class SpeechTurnSegmentation(Pipeline):
    """Combine speech activity and speaker change detections for segmentation

    Parameters
    ----------
    sad_scores : `Path` or 'oracle'
        Path to precomputed speech activity detection scores.
        Use 'oracle' to assume perfect speech activity detection.
    scd_scores : `Path` or 'oracle'
        Path to precomputed SCD scores on disk.
        Use 'oracle' to assume perfect speaker change detection.
    non_speech : `bool`
        Mark non-speech regions as speaker change. Defaults to True.
    purity : `float`, optional
        Target purity. Defaults to 0.95
    """

    def __init__(self, sad_scores: Optional[Union[Path, str]] = None,
                       scd_scores: Optional[Union[Path, str]] = None,
                       non_speech: Optional[bool] = True,
                       purity: Optional[float] = 0.95):
        super().__init__()

        self.sad_scores = sad_scores
        if self.sad_scores == 'oracle':
            self.speech_activity_detection = OracleSpeechActivityDetection()
        else:
            self.speech_activity_detection = SpeechActivityDetection(
                scores=self.sad_scores)

        self.scd_scores = scd_scores
        if self.scd_scores == 'oracle':
            self.speaker_change_detection = None
            self.speech_turn_segmentation = OracleSpeechTurnSegmentation()
        else:
            self.speaker_change_detection = SpeakerChangeDetection(
                scores=self.scd_scores)

        self.non_speech = non_speech
        self.purity = purity


    def __call__(self, current_file: dict) -> Annotation:
        """Apply speech turn segmentation

        Parameter
        ---------
        current_file : `dict`
            Dictionary as provided by `pyannote.database`.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Hypothesized speech turns
        """
        if self.scd_scores == 'oracle':
            speech_turns = self.speech_turn_segmentation(current_file)
            return speech_turns
            
        # speech regions
        sad = self.speech_activity_detection(current_file).get_timeline()

        scd = self.speaker_change_detection(current_file)
        speech_turns = scd.crop(sad, mode='intersection')

        # at this point, consecutive speech turns separated by non-speech
        # might be assigned the same label (because scd might have missed
        # speech/non-speech boundaries)

        # assign one unique label per speech turn
        if self.non_speech:
            speech_turns = speech_turns.relabel_tracks(generator='string')

        speech_turns.modality = 'speaker'
        return speech_turns

    def loss(self, current_file: dict, hypothesis: Annotation) -> float:
        """Compute (1 - coverage) at target purity

        If purity < target, return 1 + (1 - purity)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Speech turns.

        Returns
        -------
        error : `float`
            1. - cluster coverage.
        """

        metric = DiarizationPurityCoverageFMeasure()
        reference  = current_file['annotation']
        uem = get_annotated(current_file)
        f_measure = metric(reference, hypothesis, uem=uem)
        purity, coverage, _ = metric.compute_metrics()
        if purity > self.purity:
            return 1. - coverage
        else:
            return 1. + (1. - purity)
