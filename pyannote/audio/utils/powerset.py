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

# AUTHORS
# Hervé BREDIN - https://herve.niderb.fr
# Alexis PLAQUET

from functools import cached_property
from itertools import combinations

import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


class Powerset(nn.Module):
    """Powerset to multilabel conversion, and back.

    Parameters
    ----------
    num_classes : int
        Number of regular classes.
    max_set_size : int
        Maximum number of classes in each set.
    """

    def __init__(self, num_classes: int, max_set_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.max_set_size = max_set_size

        self.register_buffer("mapping", self.build_mapping(), persistent=False)
        self.register_buffer("cardinality", self.build_cardinality(), persistent=False)

    @cached_property
    def num_powerset_classes(self) -> int:
        # compute number of subsets of size at most "max_set_size"
        # e.g. with num_classes = 3 and max_set_size = 2:
        # {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
        return int(
            sum(
                scipy.special.binom(self.num_classes, i)
                for i in range(0, self.max_set_size + 1)
            )
        )

    def build_mapping(self) -> torch.Tensor:
        mapping = torch.zeros(self.num_powerset_classes, self.num_classes)
        powerset_k = 0
        for set_size in range(0, self.max_set_size + 1):
            for current_set in combinations(range(self.num_classes), set_size):
                mapping[powerset_k, current_set] = 1
                powerset_k += 1

        return mapping

    def build_cardinality(self) -> torch.Tensor:
        """Compute size of each powerset class"""
        cardinality = torch.zeros(self.num_powerset_classes)
        powerset_k = 0
        for set_size in range(0, self.max_set_size + 1):
            for _ in combinations(range(self.num_classes), set_size):
                cardinality[powerset_k] = set_size
                powerset_k += 1
        return cardinality

    def to_multilabel(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Convert predictions from powerset to multi-label

        Parameter
        ---------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            Soft predictions in "powerset" space.
        soft : bool, optional
            Return soft multi-label predictions. Defaults to False (i.e. hard predictions)
            Assumes that `powerset` are "logits" (not "probabilities").

        Returns
        -------
        multi_label : (batch_size, num_frames, num_classes) torch.Tensor
            Predictions in "multi-label" space.
        """

        if soft:
            powerset_probs = torch.exp(powerset)
        else:
            powerset_probs = torch.nn.functional.one_hot(
                torch.argmax(powerset, dim=-1),
                self.num_powerset_classes,
            ).float()

        return torch.matmul(powerset_probs, self.mapping)

    def forward(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Alias for `to_multilabel`"""
        return self.to_multilabel(powerset, soft=soft)

    def to_powerset(self, multilabel: torch.Tensor) -> torch.Tensor:
        """Convert (hard) predictions from multi-label to powerset

        Parameter
        ---------
        multi_label : (batch_size, num_frames, num_classes) torch.Tensor
            Prediction in "multi-label" space.

        Returns
        -------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            Hard, one-hot prediction in "powerset" space.

        Note
        ----
        This method will not complain if `multilabel` is provided a soft predictions
        (e.g. the output of a sigmoid-ed classifier). However, in that particular
        case, the resulting powerset output will most likely not make much sense.
        """
        return F.one_hot(
            torch.argmax(torch.matmul(multilabel, self.mapping.T), dim=-1),
            num_classes=self.num_powerset_classes,
        )
