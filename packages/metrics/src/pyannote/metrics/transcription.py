# The MIT License (MIT)

# Copyright (c) 2025- pyannoteAI

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


from __future__ import annotations

try:
    import meeteval
    from meeteval.io.seglst import SegLST
    from meeteval.wer.wer.cp import CPErrorRate
    from meeteval.wer.wer.orc import OrcErrorRate

    MEETEVAL_IS_AVAILABLE = True

except ImportError:
    MEETEVAL_IS_AVAILABLE = False

from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents

TOTAL = "total"
INSERTION = "insertion"
DELETION = "deletion"
SUBSTITUTION = "substitution"


class BaseWordErrorRate(BaseMetric):
    def validate(self, reference: SegLST, hypothesis: SegLST) -> str:
        """Validate that both reference and hypothesis are single-session

        Parameters
        ----------
        reference : SegLST
            Reference transcription segments.
        hypothesis : SegLST
            Hypothesis transcription segments.

        Returns
        -------
        session_id : str
            The single session ID found in both reference and hypothesis.
        """

        # check that reference is single session
        reference_session_ids = {s["session_id"] for s in reference}
        if len(reference_session_ids) != 1:
            raise ValueError(
                "Reference must contain exactly one session. "
                f"Found sessions: {list(reference_session_ids)}"
            )

        # check that hypothesis is single session
        hypothesis_session_ids = {s["session_id"] for s in hypothesis}
        if len(hypothesis_session_ids) != 1:
            raise ValueError(
                "Hypothesis must contain exactly one session. "
                f"Found sessions: {list(hypothesis_session_ids)}"
            )

        # check that both reference and hypothesis refer to the same single session
        if reference_session_ids != hypothesis_session_ids:
            raise ValueError(
                "Reference and hypothesis must refer to the same single session. "
                f"Found reference sessions: {list(reference_session_ids)} "
                f"and hypothesis sessions: {list(hypothesis_session_ids)}"
            )

        return reference_session_ids.pop()

    @classmethod
    def metric_components(cls) -> MetricComponents:
        """Return the list of metric components."""
        return [
            TOTAL,
            INSERTION,
            DELETION,
            SUBSTITUTION,
        ]

    def compute_metric(self, detail: Details) -> float:
        numerator = detail[INSERTION] + detail[SUBSTITUTION] + detail[DELETION]
        denominator = detail[TOTAL]
        if denominator == 0.0:
            if numerator == 0:
                return 0.0
            else:
                return 1.0
        else:
            return numerator / denominator


class WordErrorRate(BaseWordErrorRate):
    """Word Error Rate"""

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "WER"

    def __init__(self, **kwargs):
        if not MEETEVAL_IS_AVAILABLE:
            raise ImportError(
                "WordErrorRate metric is not available. "
                "Install `pyannote.metrics` with the `transcription` extra to use it."
            )

        super().__init__(**kwargs)

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        _ = self.validate(reference, hypothesis)

        ref_txt = " ".join(
            s["words"] for s in sorted(reference, key=lambda s: s["start_time"])
        )
        hyp_txt = " ".join(
            s["words"] for s in sorted(hypothesis, key=lambda s: s["start_time"])
        )

        result = meeteval.wer.siso_word_error_rate(
            [{"words": ref_txt}], [{"words": hyp_txt}]
        )

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }


WER = WordErrorRate


class ConcatenatedMinimumPermutationWordErrorRate(BaseWordErrorRate):
    """Concatenated minimum-Permutation Word Error Rate (cpWER)."""

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "cpWER"

    def __init__(self, **kwargs):
        if not MEETEVAL_IS_AVAILABLE:
            raise ImportError(
                "ConcatenatedMinimumPermutationWordErrorRate metric is not available. "
                "Install `pyannote.metrics` with the `transcription` extra to use it."
            )

        super().__init__(**kwargs)

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        session_id = self.validate(reference, hypothesis)

        # compute concatenated minimum-permutation WER
        result: CPErrorRate = meeteval.wer.cpwer(
            reference,
            hypothesis,
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }


class TimeConstrainedMinimumPermutationWordErrorRate(BaseWordErrorRate):
    """Time-Constrained minimum-Permutation Word Error Rate (tcpWER).

    Parameters
    ----------
    collar : float, optional
        Collar applied to hypothesis pseudo-word level timings, in seconds.
        Defaults to 5 seconds.
    """

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "tcpWER"

    def __init__(self, collar: float = 5.0, **kwargs):
        if not MEETEVAL_IS_AVAILABLE:
            raise ImportError(
                "TimeConstrainedMinimumPermutationWordErrorRate metric is not available. "
                "Install `pyannote.metrics` with the `transcription` extra to use it."
            )

        super().__init__(**kwargs)
        self.collar = collar

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        session_id = self.validate(reference, hypothesis)

        # compute time-constrained minimum-permutation WER
        result: CPErrorRate = meeteval.wer.tcpwer(
            reference, hypothesis, collar=self.collar
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }


class TimeConstrainedOptimalReferenceCombinationWordErrorRate(BaseWordErrorRate):
    """Time-Constrained Optimal Reference Combination Word Error Rate (tcorcWER)

    Parameters
    ----------
    collar : float, optional
        Collar applied to hypothesis pseudo-word level timings, in seconds.
        Defaults to 5 seconds.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "tcorcWER"

    def __init__(self, collar: float = 5.0, **kwargs):
        if not MEETEVAL_IS_AVAILABLE:
            raise ImportError(
                "TimeConstrainedOptimalReferenceCombinationWordErrorRate metric is not available. "
                "Install `pyannote.metrics` with the `transcription` extra to use it."
            )

        super().__init__(**kwargs)
        self.collar = collar

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        session_id = self.validate(reference, hypothesis)

        # compute time-constrained minimum-permutation WER
        result: OrcErrorRate = meeteval.wer.tcorcwer(
            reference, hypothesis, collar=self.collar
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }
