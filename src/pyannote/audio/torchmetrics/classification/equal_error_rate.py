# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional

import torch
from pyannote.metrics.binary_classification import det_curve
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class EqualErrorRate(Metric):

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, distances: bool = True, compute_on_cpu: bool = True, **kwargs):
        super().__init__(compute_on_cpu=compute_on_cpu, **kwargs)
        self.distances = distances
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")

    def update(self, scores: torch.Tensor, y_true: torch.Tensor) -> None:
        self.scores.append(scores)
        self.y_true.append(y_true)

    def compute(self) -> torch.Tensor:
        scores = dim_zero_cat(self.scores)
        y_true = dim_zero_cat(self.y_true)
        _, _, _, eer = det_curve(y_true.cpu(), scores.cpu(), distances=self.distances)
        return torch.tensor(eer)
