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

from scipy.optimize import minimize_scalar
from pandas import DataFrame
from pyannote.metrics.base import BaseMetric
from pyannote.core import Annotation


class CollarOptimizer:
    """Optimize the value used for collar on the specified metric.
    Wrapper of `scipy.optimize.minimize_scalar`.

    Parameters
    ----------
    metric: one of pyannote.metric
        The metric upon which optimize.
    bounds: tuple, optional
        Interval search for the collar value, passed as (start, end).
    method: str
        method used for the optimization. Must be one of the methods
        accepted by `scipy.optimize.minimize_scalar`.

    Usage
    -----
    >>> uri = file["uri"]
    >>> prediction = SpeakerDiarization(file)
    >>> annotation = file["annotation"]
    >>> metric = DiarizationErrorRate()
    >>> collar_optimizer = CollarOptimizer(metric)
    >>> collar_optimizer[uri] = {"prediction": prediction, "annotation": annotation}
    >>> best_collar, best_der = collar_optimizer.optimize()

    Notes
    -----
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
    for more details about `scipy.optimize.minimize_scalar`
    """

    def __init__(
        self,
        metric: BaseMetric,
        bounds: tuple[float, float] = (0., 1.),
        method: str = "Bounded",
    ):
        self.metric = metric
        self.bounds = bounds
        self.method = method

        self._cached_data = {}
        self._dataframes: dict[float, DataFrame] = {}

    def __setitem__(self, uri, value: dict):
        self._cached_data[uri] = {k: v for k, v in value.items()}

    def _compute_metric(self, collar: float) -> float:
        self.metric.reset()
        collar = round(collar, 3)

        for uri in self._cached_data:
            prediction: Annotation = self._cached_data[uri]["prediction"]
            prediction = prediction.support(collar=collar)

            uem = self._cached_data[uri].get("annotated", None)

            _ = self.metric(
                prediction,
                self._cached_data[uri]["annotation"],
                uem=uem,
            )

        self._dataframes[collar] = self.metric.report()

        return abs(self.metric)

    def optimize(self) -> tuple[float, DataFrame]:
        """Optimize collar value for `self.metric`

        Returns
        -------
        best_collar : float
            optimized collar value.
        metric_val: pandas.DataFrame
            best metric values with the optimized collar.
        """
        res = minimize_scalar(
            self._compute_metric, bounds=self.bounds, method=self.method
        )

        best_collar = round(float(res.x), 3)
        metric_df = self._dataframes[best_collar]

        return best_collar, metric_df
