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
from itertools import combinations, permutations
from typing import Dict, Tuple

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
        """Compute powerset to regular mapping

        Returns
        -------
        mapping : (num_powerset_classes, num_classes) torch.Tensor
            mapping[i, j] == 1 if jth regular class is a member of ith powerset class
            mapping[i, j] == 0 otherwise

        Example
        -------
        With num_classes == 3 and max_set_size == 2, returns

            [0, 0, 0]  # none
            [1, 0, 0]  # class #1
            [0, 1, 0]  # class #2
            [0, 0, 1]  # class #3
            [1, 1, 0]  # classes #1 and #2
            [1, 0, 1]  # classes #1 and #3
            [0, 1, 1]  # classes #2 and #3

        """
        mapping = torch.zeros(self.num_powerset_classes, self.num_classes)
        powerset_k = 0
        for set_size in range(0, self.max_set_size + 1):
            for current_set in combinations(range(self.num_classes), set_size):
                mapping[powerset_k, current_set] = 1
                powerset_k += 1

        return mapping

    def build_cardinality(self) -> torch.Tensor:
        """Compute size of each powerset class"""
        return torch.sum(self.mapping, dim=1)

    def to_multilabel(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Convert predictions from powerset to multi-label

        Parameter
        ---------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            Soft predictions in "powerset" space.
        soft : bool, optional
            Return soft multi-label predictions. Defaults to False (i.e. hard predictions)
            Assumes that `powerset` are "log probabilities".

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

    def _permutation_powerset(
        self, multilabel_permutation: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Helper function for `permutation_mapping` property

        Takes a (num_classes,)-shaped permutation in multilabel space and returns
        the corresponding (num_powerset_classes,)-shaped permutation in powerset space.
        This does not cache anything and only works on one single permutation at a time.

        Parameters
        ----------
        multilabel_permutation : tuple of int
            Permutation in multilabel space.

        Returns
        -------
        powerset_permutation : tuple of int
            Permutation in powerset space.

        Example
        -------
        >>> powerset = Powerset(3, 2)
        >>> powerset._permutation_powerset((1, 0, 2))
        # (0, 2, 1, 3, 4, 6, 5)

        """

        permutated_mapping: torch.Tensor = self.mapping[:, multilabel_permutation]

        arange = torch.arange(
            self.num_classes, device=self.mapping.device, dtype=torch.int
        )
        powers_of_two = (2**arange).tile((self.num_powerset_classes, 1))

        # compute the encoding of the powerset classes in this 2**N space, before and after
        # permutation of the columns (mapping cols=labels, mapping rows=powerset classes)
        before = torch.sum(self.mapping * powers_of_two, dim=-1)
        after = torch.sum(permutated_mapping * powers_of_two, dim=-1)

        # find before-to-after permutation
        powerset_permutation = (before[None] == after[:, None]).int().argmax(dim=0)

        # return as tuple of indices
        return tuple(powerset_permutation.tolist())

    @cached_property
    def permutation_mapping(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """Mapping between multilabel and powerset permutations

        Example
        -------
        With num_classes == 3 and max_set_size == 2, returns

        {
            (0, 1, 2): (0, 1, 2, 3, 4, 5, 6),
            (0, 2, 1): (0, 1, 3, 2, 5, 4, 6),
            (1, 0, 2): (0, 2, 1, 3, 4, 6, 5),
            (1, 2, 0): (0, 2, 3, 1, 6, 4, 5),
            (2, 0, 1): (0, 3, 1, 2, 5, 6, 4),
            (2, 1, 0): (0, 3, 2, 1, 6, 5, 4)
        }
        """
        permutation_mapping = {}

        for multilabel_permutation in permutations(
            range(self.num_classes), self.num_classes
        ):
            permutation_mapping[
                tuple(multilabel_permutation)
            ] = self._permutation_powerset(multilabel_permutation)

        return permutation_mapping
