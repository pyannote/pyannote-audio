# The MIT License (MIT)

# Copyright (c) 2012-2020 CNRS

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
# Marvin LAVECHIN

from pyannote.core import Annotation, Timeline

from .base import BaseMetric, f_measure
from .types import Details, MetricComponents
from .utils import UEMSupportMixin

DER_NAME = "detection error rate"
DER_TOTAL = "total"
DER_FALSE_ALARM = "false alarm"
DER_MISS = "miss"


class DetectionErrorRate(UEMSupportMixin, BaseMetric):
    """Detection error rate

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as (fa + miss) / total, where fa is the duration of false
    alarm (e.g. non-speech classified as speech), miss is the duration of
    missed detection (e.g. speech classified as non-speech), and total is the
    total duration of the positive class in the reference.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls) -> str:
        return DER_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [DER_TOTAL, DER_FALSE_ALARM, DER_MISS]

    def __init__(self, collar: float = 0.0, skip_overlap: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[DER_MISS] = false_negative
        detail[DER_FALSE_ALARM] = false_positive
        detail[DER_TOTAL] = reference.duration()

        return detail

    def compute_metric(self, detail: Details) -> float:
        error = 1.0 * (detail[DER_FALSE_ALARM] + detail[DER_MISS])
        total = 1.0 * detail[DER_TOTAL]
        if total == 0.0:
            if error == 0:
                return 0.0
            else:
                return 1.0
        else:
            return error / total


ACCURACY_NAME = "detection accuracy"
ACCURACY_TRUE_POSITIVE = "true positive"
ACCURACY_TRUE_NEGATIVE = "true negative"
ACCURACY_FALSE_POSITIVE = "false positive"
ACCURACY_FALSE_NEGATIVE = "false negative"


class DetectionAccuracy(DetectionErrorRate):
    """Detection accuracy

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as (tp + tn) / total, where tp is the duration of true
    positive (e.g. speech classified as speech), tn is the duration of true
    negative (e.g. non-speech classified as non-speech), and total is the total
    duration of the input signal.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return ACCURACY_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            ACCURACY_TRUE_POSITIVE,
            ACCURACY_TRUE_NEGATIVE,
            ACCURACY_FALSE_POSITIVE,
            ACCURACY_FALSE_NEGATIVE,
        ]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        true_negative = 0.0
        for r_, h_ in reference_.co_iter(hypothesis_):
            true_negative += (r_ & h_).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[ACCURACY_TRUE_NEGATIVE] = true_negative
        detail[ACCURACY_TRUE_POSITIVE] = true_positive
        detail[ACCURACY_FALSE_NEGATIVE] = false_negative
        detail[ACCURACY_FALSE_POSITIVE] = false_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        numerator = 1.0 * (
            detail[ACCURACY_TRUE_NEGATIVE] + detail[ACCURACY_TRUE_POSITIVE]
        )
        denominator = 1.0 * (
            detail[ACCURACY_TRUE_NEGATIVE]
            + detail[ACCURACY_TRUE_POSITIVE]
            + detail[ACCURACY_FALSE_NEGATIVE]
            + detail[ACCURACY_FALSE_POSITIVE]
        )

        if denominator == 0.0:
            return 1.0
        else:
            return numerator / denominator


PRECISION_NAME = "detection precision"
PRECISION_RETRIEVED = "retrieved"
PRECISION_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionPrecision(DetectionErrorRate):
    """Detection precision

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as tp / (tp + fp), where tp is the duration of true positive
    (e.g. speech classified as speech), and fp is the duration of false
    positive (e.g. non-speech classified as speech).

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        detail = {}
        detail[PRECISION_RETRIEVED] = true_positive + false_positive
        detail[PRECISION_RELEVANT_RETRIEVED] = true_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        relevant_retrieved = 1.0 * detail[PRECISION_RELEVANT_RETRIEVED]
        retrieved = 1.0 * detail[PRECISION_RETRIEVED]
        if retrieved == 0.0:
            return 1.0
        else:
            return relevant_retrieved / retrieved


RECALL_NAME = "detection recall"
RECALL_RELEVANT = "relevant"
RECALL_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionRecall(DetectionErrorRate):
    """Detection recall

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection, for instance. Inputs are expected to only
    contain segments corresponding to the positive class (e.g. speech regions).
    Gaps in the inputs considered as the negative class (e.g. non-speech
    regions).

    It is computed as tp / (tp + fn), where tp is the duration of true positive
    (e.g. speech classified as speech), and fn is the duration of false
    negative (e.g. speech classified as non-speech).

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        hypothesis_ = hypothesis.gaps(support=uem)

        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {}
        detail[RECALL_RELEVANT] = true_positive + false_negative
        detail[RECALL_RELEVANT_RETRIEVED] = true_positive

        return detail

    def compute_metric(self, detail: Details) -> float:
        relevant_retrieved = 1.0 * detail[RECALL_RELEVANT_RETRIEVED]
        relevant = 1.0 * detail[RECALL_RELEVANT]
        if relevant == 0.0:
            if relevant_retrieved == 0:
                return 1.0
            else:
                return 0.0
        else:
            return relevant_retrieved / relevant


DFS_NAME = "F[precision|recall]"
DFS_PRECISION_RETRIEVED = "retrieved"
DFS_RECALL_RELEVANT = "relevant"
DFS_RELEVANT_RETRIEVED = "relevant retrieved"


class DetectionPrecisionRecallFMeasure(UEMSupportMixin, BaseMetric):
    """Compute detection precision and recall, and return their F-score

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    beta : float, optional
        When beta > 1, greater importance is given to recall.
        When beta < 1, greater importance is given to precision.
        Defaults to 1.

    See also
    --------
    pyannote.metrics.detection.DetectionPrecision
    pyannote.metrics.detection.DetectionRecall
    pyannote.metrics.base.f_measure

    """

    @classmethod
    def metric_name(cls):
        return DFS_NAME

    @classmethod
    def metric_components(cls):
        return [DFS_PRECISION_RETRIEVED, DFS_RECALL_RELEVANT, DFS_RELEVANT_RETRIEVED]

    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.beta = beta

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        # Better to recompute everything from scratch instead of calling the
        # DetectionPrecision & DetectionRecall classes (we skip one of the loop
        # that computes the amount of true positives).
        true_positive = 0.0
        for r, h in reference.co_iter(hypothesis):
            true_positive += (r & h).duration

        false_positive = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            false_positive += (r_ & h).duration

        false_negative = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            false_negative += (r & h_).duration

        detail = {
            DFS_PRECISION_RETRIEVED: true_positive + false_positive,
            DFS_RECALL_RELEVANT: true_positive + false_negative,
            DFS_RELEVANT_RETRIEVED: true_positive,
        }

        return detail

    def compute_metric(self, detail: Details) -> float:
        _, _, value = self.compute_metrics(detail=detail)
        return value

    def compute_metrics(
        self, detail: Details | None = None
    ) -> tuple[float, float, float]:

        detail = self.accumulated_ if detail is None else detail
        precision_retrieved = detail[DFS_PRECISION_RETRIEVED]
        recall_relevant = detail[DFS_RECALL_RELEVANT]
        relevant_retrieved = detail[DFS_RELEVANT_RETRIEVED]

        # Special cases : precision
        if precision_retrieved == 0.0:
            precision = 1
        else:
            precision = relevant_retrieved / precision_retrieved

        # Special cases : recall
        if recall_relevant == 0.0:
            if relevant_retrieved == 0:
                recall = 1.0
            else:
                recall = 0.0
        else:
            recall = relevant_retrieved / recall_relevant

        return precision, recall, f_measure(precision, recall, beta=self.beta)


DCF_NAME = "detection cost function"
DCF_POS_TOTAL = "positive class total"  # Total duration of positive class.
DCF_NEG_TOTAL = "negative class total"  # Total duration of negative class.
DCF_FALSE_ALARM = "false alarm"  # Total duration of false alarms.
DCF_MISS = "miss"  # Total duration of misses.


class DetectionCostFunction(UEMSupportMixin, BaseMetric):
    """Detection cost function.

    This metric can be used to evaluate binary classification tasks such as
    speech activity detection. Inputs are expected to only contain segments
    corresponding to the positive class (e.g. speech regions). Gaps in the
    inputs considered as the negative class (e.g. non-speech regions).

    Detection cost function (DCF), as defined by NIST for OpenSAT 2019, is
    0.25*far + 0.75*missr, where far is the false alarm rate
    (i.e., the proportion of non-speech incorrectly classified as speech)
    and missr is the miss rate (i.e., the proportion of speech incorrectly
    classified as non-speech.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments (one half before, one half after).
        Defaults to 0.0.

    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    fa_weight : float, optional
        Weight for false alarm rate.
        Defaults to 0.25.

    miss_weight : float, optional
        Weight for miss rate.
        Defaults to 0.75.

    kwargs
        Keyword arguments passed to :class:`pyannote.metrics.base.BaseMetric`.

    References
    ----------
    "OpenSAT19 Evaluation Plan v2." https://www.nist.gov/system/files/documents/2018/11/05/opensat19_evaluation_plan_v2_11-5-18.pdf
    """

    def __init__(
        self, collar=0.0, skip_overlap=False, fa_weight=0.25, miss_weight=0.75, **kwargs
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.fa_weight = fa_weight
        self.miss_weight = miss_weight

    @classmethod
    def metric_name(cls):
        return DCF_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [DCF_POS_TOTAL, DCF_NEG_TOTAL, DCF_FALSE_ALARM, DCF_MISS]

    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        **kwargs,
    ) -> Details:

        reference, hypothesis, uem = self.uemify(
            reference,
            hypothesis,
            uem=uem,
            collar=self.collar,
            skip_overlap=self.skip_overlap,
            returns_uem=True,
        )

        # Obtain timelines corresponding to positive class.
        reference = reference.get_timeline(copy=False).support()
        hypothesis = hypothesis.get_timeline(copy=False).support()

        # Obtain timelines corresponding to negative class.
        reference_ = reference.gaps(support=uem)
        hypothesis_ = hypothesis.gaps(support=uem)

        # Compute total positive/negative durations.
        pos_dur = reference.duration()
        neg_dur = reference_.duration()

        # Compute total miss duration.
        miss_dur = 0.0
        for r, h_ in reference.co_iter(hypothesis_):
            miss_dur += (r & h_).duration

        # Compute total false alarm duration.
        fa_dur = 0.0
        for r_, h in reference_.co_iter(hypothesis):
            fa_dur += (r_ & h).duration

        components = {
            DCF_POS_TOTAL: pos_dur,
            DCF_NEG_TOTAL: neg_dur,
            DCF_MISS: miss_dur,
            DCF_FALSE_ALARM: fa_dur,
        }

        return components

    def compute_metric(self, components: Details) -> float:
        def _compute_rate(num, denom):
            if denom == 0.0:
                if num == 0.0:
                    return 0.0
                return 1.0
            return num / denom

        # Compute false alarm rate.
        neg_dur = components[DCF_NEG_TOTAL]
        fa_dur = components[DCF_FALSE_ALARM]
        fa_rate = _compute_rate(fa_dur, neg_dur)

        # Compute miss rate.
        pos_dur = components[DCF_POS_TOTAL]
        miss_dur = components[DCF_MISS]
        miss_rate = _compute_rate(miss_dur, pos_dur)

        return self.fa_weight * fa_rate + self.miss_weight * miss_rate
