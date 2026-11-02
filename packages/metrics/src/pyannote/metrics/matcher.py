# The MIT License (MIT)

# Copyright (c) 2012-2019 CNRS

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
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from pyannote.core import Annotation
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from pyannote.core.utils.types import Label

MATCH_CORRECT = "correct"
MATCH_CONFUSION = "confusion"
MATCH_MISSED_DETECTION = "missed detection"
MATCH_FALSE_ALARM = "false alarm"
MATCH_TOTAL = "total"


class LabelMatcher:
    """
    ID matcher base class mixin.

    All ID matcher classes must inherit from this class and implement
    .match() -- ie return True if two IDs match and False
    otherwise.
    """

    def match(self, rlabel: "Label", hlabel: "Label") -> bool:
        """
        Parameters
        ----------
        rlabel :
            Reference label
        hlabel :
            Hypothesis label

        Returns
        -------
        match : bool
            True if labels match, False otherwise.

        """
        # Two IDs match if they are equal to each other
        return rlabel == hlabel

    def __call__(
        self, rlabels: Iterable["Label"], hlabels: Iterable["Label"]
    ) -> tuple[dict[str, int], dict[str, list["Label"]]]:
        """

        Parameters
        ----------
        rlabels, hlabels : iterable
            Reference and hypothesis labels

        Returns
        -------
        counts : dict
        details : dict

        """

        # counts and details
        counts = {
            MATCH_CORRECT: 0,
            MATCH_CONFUSION: 0,
            MATCH_MISSED_DETECTION: 0,
            MATCH_FALSE_ALARM: 0,
            MATCH_TOTAL: 0,
        }

        details = {
            MATCH_CORRECT: [],
            MATCH_CONFUSION: [],
            MATCH_MISSED_DETECTION: [],
            MATCH_FALSE_ALARM: [],
        }
        # this is to make sure rlabels and hlabels are lists
        # as we will access them later by index
        rlabels = list(rlabels)
        hlabels = list(hlabels)

        NR = len(rlabels)
        NH = len(hlabels)
        N = max(NR, NH)

        # corner case
        if N == 0:
            return counts, details

        # initialize match matrix
        # with True if labels match and False otherwise
        match = np.zeros((N, N), dtype=bool)
        for r, rlabel in enumerate(rlabels):
            for h, hlabel in enumerate(hlabels):
                match[r, h] = self.match(rlabel, hlabel)

        # find one-to-one mapping that maximize total number of matches
        # using the Hungarian algorithm and computes error accordingly
        for r, h in zip(*linear_sum_assignment(~match)):
            # hypothesis label is matched with unexisting reference label
            # ==> this is a false alarm
            if r >= NR:
                counts[MATCH_FALSE_ALARM] += 1
                details[MATCH_FALSE_ALARM].append(hlabels[h])

            # reference label is matched with unexisting hypothesis label
            # ==> this is a missed detection
            elif h >= NH:
                counts[MATCH_MISSED_DETECTION] += 1
                details[MATCH_MISSED_DETECTION].append(rlabels[r])

            # reference and hypothesis labels match
            # ==> this is a correct detection
            elif match[r, h]:
                counts[MATCH_CORRECT] += 1
                details[MATCH_CORRECT].append((rlabels[r], hlabels[h]))

            # reference and hypothesis do not match
            # ==> this is a confusion
            else:
                counts[MATCH_CONFUSION] += 1
                details[MATCH_CONFUSION].append((rlabels[r], hlabels[h]))

        counts[MATCH_TOTAL] += NR

        # returns counts and details
        return counts, details


class HungarianMapper:
    def __call__(self, A: Annotation, B: Annotation) -> dict["Label", "Label"]:
        mapping = {}

        cooccurrence = A * B
        a_labels, b_labels = A.labels(), B.labels()

        for a, b in zip(*linear_sum_assignment(-cooccurrence)):
            if cooccurrence[a, b] > 0:
                mapping[a_labels[a]] = b_labels[b]

        return mapping


class GreedyMapper:
    def __call__(self, A: Annotation, B: Annotation) -> dict["Label", "Label"]:
        mapping = {}

        cooccurrence = A * B
        Na, Nb = cooccurrence.shape
        a_labels, b_labels = A.labels(), B.labels()

        for i in range(min(Na, Nb)):
            a, b = np.unravel_index(np.argmax(cooccurrence), (Na, Nb))

            if cooccurrence[a, b] > 0:
                mapping[a_labels[a]] = b_labels[b]
                cooccurrence[a, :] = 0.0
                cooccurrence[:, b] = 0.0
                continue

            break

        return mapping
