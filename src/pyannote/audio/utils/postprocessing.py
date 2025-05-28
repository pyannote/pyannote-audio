# The MIT License (MIT)
#
# Copyright (c) 2025- pyannoteAI
#
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

from functools import partial
from scipy.optimize import minimize_scalar
from pandas import DataFrame
from pyannote.metrics.base import BaseMetric


class MinDurationOffOptimizer:
    """Utility to optimize `min_duration_off` value for a given metric."""

    def _compute_metric(self, files, metric, collar: float) -> float:
        metric.reset()
        for file in files:
            _ = metric(
                file["annotation"],
                file["speaker_diarization"].support(collar=collar),
                uem=file["annotated"],
            )
        self._reports[collar] = metric.report()
        return abs(metric)

    def __call__(self, files, metric: BaseMetric) -> tuple[float, DataFrame]:
        """Optimize 'min_duration_off' value for `metric`

        Parameters
        ----------
        files : list[dict]
            List of dictionaries containing 'uri', 'annotation', and 'annotated' keys.
            Each dictionary represents a file with its corresponding annotation and UEM.
        metric : BaseMetric
            Metric to optimize against. It should be a subclass of `BaseMetric`.
            
        Returns
        -------
        best_min_duration_off : float
            Optimize min_duration_off parameter.
        best_report: pandas.DataFrame
            Corresponding report.
        """

        self._reports: dict[float, DataFrame] = dict()

        res = minimize_scalar(
            partial(self._compute_metric, files, metric), bounds=(0.0, 1.0), method="Bounded"
        )

        best_min_duration_off = float(res.x)

        return best_min_duration_off, self._reports[best_min_duration_off]
