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
# Hervé BREDIN - http://herve.niderb.fr

import sys
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike
from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline

from .base import BaseMetric
from .binary_classification import det_curve
from .types import Details, MetricComponents

SPOTTING_TARGET = "target"
SPOTTING_SPK_LATENCY = "speaker_latency"
SPOTTING_SPK_SCORE = "spk_score"
SPOTTING_ABS_LATENCY = "absolute_latency"
SPOTTING_ABS_SCORE = "abs_score"
SPOTTING_SCORE = "score"


class LowLatencySpeakerSpotting(BaseMetric):
    """Evaluation of low-latency speaker spotting (LLSS) systems

    LLSS systems can be evaluated in two ways: with fixed or variable latency.

    * When latency is fixed a priori (default), only scores reported by the
    system within the requested latency range are considered. Varying the
    detection threshold has no impact on the actual latency of the system. It
    only impacts the detection performance.

    * In variable latency mode, the whole stream of scores is considered.
    Varying the detection threshold will impact both the detection performance
    and the detection latency. Each trial will result in the alarm being
    triggered with a different latency. In case the alarm is not triggered at
    all (missed detection), the latency is arbitrarily set to the value one
    would obtain if it were triggered at the end of the last target speech
    turn. The reported latency is the average latency over all target trials.

    Parameters
    ----------
    latencies : float iterable, optional
        Switch to fixed latency mode, using provided `latencies`.
        Defaults to [1, 5, 10, 30, 60] (in seconds).
    thresholds : float iterable, optional
        Switch to variable latency mode, using provided detection `thresholds`.
        Defaults to fixed latency mode.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "Low-latency speaker spotting"

    def metric_components(self) -> dict[str, float]:
        return {"target": 0.0}

    def __init__(
        self, thresholds: ArrayLike | None = None, latencies: ArrayLike | None = None
    ):
        super().__init__()

        if thresholds is None and latencies is None:
            latencies = [1, 5, 10, 30, 60]

        if thresholds is not None and latencies is not None:
            raise ValueError("One must choose between fixed and variable latency.")

        if thresholds is not None:
            self.thresholds = np.sort(thresholds)

        if latencies is not None:
            latencies = np.sort(latencies)

        self.latencies = latencies

    def compute_metric(self, detail: MetricComponents):
        return None

    def _fixed_latency(
        self, reference: Timeline, timestamps: list[float], scores: list[float]
    ) -> Details:

        if not reference:
            target_trial = False
            spk_score = np.max(scores) * np.ones((len(self.latencies), 1))
            abs_score = spk_score

        else:
            target_trial = True

            # cumulative target speech duration after each speech turn
            total = np.cumsum([segment.duration for segment in reference])

            # maximum score in timerange [0, t]
            # where t is when latency is reached
            spk_score = []
            abs_score = []

            # index of speech turn when given latency is reached
            for i, latency in zip(
                np.searchsorted(total, self.latencies), self.latencies
            ):
                # maximum score in timerange [0, t]
                # where t is when latency is reached
                try:
                    t = reference[i].end - (total[i] - latency)
                    up_to = np.searchsorted(timestamps, t)
                    if up_to < 1:
                        s = -sys.float_info.max
                    else:
                        s = np.max(scores[:up_to])
                except IndexError:
                    s = np.max(scores)
                spk_score.append(s)

                # maximum score in timerange [0, t + latency]
                # where t is when target speaker starts speaking
                t = reference[0].start + latency

                up_to = np.searchsorted(timestamps, t)
                if up_to < 1:
                    s = -sys.float_info.max
                else:
                    s = np.max(scores[:up_to])
                abs_score.append(s)

            spk_score = np.array(spk_score).reshape((-1, 1))
            abs_score = np.array(abs_score).reshape((-1, 1))

        return {
            SPOTTING_TARGET: target_trial,
            SPOTTING_SPK_LATENCY: self.latencies,
            SPOTTING_SCORE: spk_score,
            SPOTTING_ABS_LATENCY: self.latencies,
            SPOTTING_ABS_SCORE: abs_score,
        }

    def _variable_latency(
        self,
        reference: Timeline | Annotation,
        timestamps: list[float],
        scores: list[float],
        **kwargs,
    ) -> Details:

        # pre-compute latencies
        speaker_latency = np.nan * np.ones((len(timestamps), 1))
        absolute_latency = np.nan * np.ones((len(timestamps), 1))
        if isinstance(reference, Annotation):
            reference = reference.get_timeline(copy=False)
        if reference:
            first_time = reference[0].start
            for i, t in enumerate(timestamps):
                so_far = Segment(first_time, t)
                speaker_latency[i] = reference.crop(so_far).duration()
                absolute_latency[i] = max(0, so_far.duration)
            # TODO | speed up latency pre-computation

        # for every threshold, compute when (if ever) alarm is triggered
        maxcum = (np.maximum.accumulate(scores)).reshape((-1, 1))
        triggered = maxcum > self.thresholds
        indices = np.array(
            [
                np.searchsorted(triggered[:, i], True)
                for i, _ in enumerate(self.thresholds)
            ]
        )

        if reference:
            target_trial = True

            absolute_latency = np.take(absolute_latency, indices, mode="clip")
            speaker_latency = np.take(speaker_latency, indices, mode="clip")

            # is alarm triggered at all?
            positive = triggered[-1, :]

            # in case alarm is not triggered, set absolute latency to duration
            # between first and last speech turn of the target speaker...
            absolute_latency[~positive] = reference.extent().duration

            # ...and set speaker latency to target's total speech duration
            speaker_latency[~positive] = reference.duration()

        else:
            target_trial = False

            # the notion of "latency" is not applicable to non-target trials
            absolute_latency = np.nan
            speaker_latency = np.nan

        return {
            SPOTTING_TARGET: target_trial,
            SPOTTING_ABS_LATENCY: absolute_latency,
            SPOTTING_SPK_LATENCY: speaker_latency,
            SPOTTING_SCORE: np.max(scores),
        }

    def compute_components(
        self,
        reference: Timeline | Annotation,
        hypothesis: SlidingWindowFeature | Iterable[tuple[float, float]],
        **kwargs,
    ) -> Details:
        """

        Parameters
        ----------
        reference : Timeline or Annotation
        hypothesis : SlidingWindowFeature or (time, score) iterable
        """

        if isinstance(hypothesis, SlidingWindowFeature):
            hypothesis = [(window.end, value) for window, value in hypothesis]
        timestamps, scores = zip(*hypothesis)

        if self.latencies is None:
            return self._variable_latency(reference, timestamps, scores)

        else:
            return self._fixed_latency(reference, timestamps, scores)

    @property
    def absolute_latency(self):
        latencies = [
            trial[SPOTTING_ABS_LATENCY] for _, trial in self if trial[SPOTTING_TARGET]
        ]
        return np.nanmean(latencies, axis=0)

    @property
    def speaker_latency(self):
        latencies = [
            trial[SPOTTING_SPK_LATENCY] for _, trial in self if trial[SPOTTING_TARGET]
        ]
        return np.nanmean(latencies, axis=0)

    # TODO : figure out return type
    def det_curve(
        self,
        cost_miss: float = 100,
        cost_fa: float = 1,
        prior_target: float = 0.01,
        return_latency: bool = False,
    ):
        """DET curve

        Parameters
        ----------
        cost_miss : float, optional
            Cost of missed detections. Defaults to 100.
        cost_fa : float, optional
            Cost of false alarms. Defaults to 1.
        prior_target : float, optional
            Target trial prior. Defaults to 0.5.
        return_latency : bool, optional
            Set to True to return latency.
            Has no effect when latencies are given at initialization time.

        Returns
        -------
        thresholds : numpy array
            Detection thresholds
        fpr : numpy array
            False alarm rate
        fnr : numpy array
            False rejection rate
        eer : float
            Equal error rate
        cdet : numpy array
            Cdet cost function
        speaker_latency : numpy array
        absolute_latency : numpy array
            Speaker and absolute latency when return_latency is set to True.
        """

        if self.latencies is None:
            y_true = np.array([trial[SPOTTING_TARGET] for _, trial in self])
            scores = np.array([trial[SPOTTING_SCORE] for _, trial in self])
            fpr, fnr, thresholds, eer = det_curve(y_true, scores, distances=False)
            fpr, fnr, thresholds = fpr[::-1], fnr[::-1], thresholds[::-1]
            cdet = cost_miss * fnr * prior_target + cost_fa * fpr * (1.0 - prior_target)

            if return_latency:
                # needed to align the thresholds used in the DET curve
                # with (self.)thresholds used to compute latencies.
                indices = np.searchsorted(thresholds, self.thresholds, side="left")

                thresholds = np.take(thresholds, indices, mode="clip")
                fpr = np.take(fpr, indices, mode="clip")
                fnr = np.take(fnr, indices, mode="clip")
                cdet = np.take(cdet, indices, mode="clip")
                return (
                    thresholds,
                    fpr,
                    fnr,
                    eer,
                    cdet,
                    self.speaker_latency,
                    self.absolute_latency,
                )

            else:
                return thresholds, fpr, fnr, eer, cdet

        else:
            y_true = np.array([trial[SPOTTING_TARGET] for _, trial in self])
            spk_scores = np.array([trial[SPOTTING_SPK_SCORE] for _, trial in self])
            abs_scores = np.array([trial[SPOTTING_ABS_SCORE] for _, trial in self])

            result = {}
            for key, scores in {"speaker": spk_scores, "absolute": abs_scores}.items():
                result[key] = {}

                for i, latency in enumerate(self.latencies):
                    fpr, fnr, theta, eer = det_curve(
                        y_true, scores[:, i], distances=False
                    )
                    fpr, fnr, theta = fpr[::-1], fnr[::-1], theta[::-1]
                    cdet = cost_miss * fnr * prior_target + cost_fa * fpr * (
                        1.0 - prior_target
                    )
                    result[key][latency] = theta, fpr, fnr, eer, cdet

            return result
