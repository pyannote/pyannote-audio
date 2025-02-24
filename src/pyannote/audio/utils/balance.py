# MIT License
#
# Copyright (c) 2025 CNRS
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
# Alexis PLAQUET

import itertools
from typing import Any, Iterable

ConditionValue = str | int | float | None
WeightingRulesDict = (
    None | dict[ConditionValue | Iterable[ConditionValue], float | tuple[float, int]]
)
InternalWeightingRulesDict = dict[Iterable[ConditionValue], tuple[float, int]]


def _get_tuples_matching_power(tuple1: tuple[Any, ...], tuple2: tuple[Any, ...]) -> int:
    """How much tuple1 matches tuple2. 0 = no match. 1=1 matching elt, etc
    None elements are not counted but pass as matched.
    e.g. (None,2,3,None) & (1,2,3,4) have a matching power of 2.
    """
    if tuple1 == tuple2:
        return len(tuple1)

    matching_elements: int = 0
    for i in range(len(tuple1)):
        if tuple1[i] == tuple2[i]:
            matching_elements += 1
        elif tuple1[i] is None:
            continue
        else:
            return 0
    return matching_elements


def _to_tuple(
    condition: ConditionValue | Iterable[ConditionValue],
) -> tuple[ConditionValue]:
    """Make sure the input str gets converted to an iterable of str (i.e. (str,))"""
    if isinstance(condition, str) or not isinstance(condition, Iterable):
        return (condition,)
    else:
        return tuple(condition)


def _raise_weighting_rule_key(val: Any, balance: Iterable[str]) -> None:
    """Raise error if the input value is not a valid weighting rule key."""
    if not isinstance(val, Iterable):
        raise ValueError(f"{val} is not a valid rule ley (not an iterable)", val)
    elif len(val) > len(balance):
        raise ValueError(
            f"{val} is not a valid rule key ({len(val)} keys vs {len(balance)} available).",
            val,
        )
    # TODO: check that each value is of a correct type (ConditionValue)


def _raise_weighting_rule_value(val: Any) -> None:
    """Raise error if the input value is not a valid weighting value."""
    if not isinstance(val, tuple) or len(val) != 2:
        raise ValueError(f"{val} is not a tuple (weight, priority)")
    if not isinstance(val[0], float) or not isinstance(val[1], int):
        raise ValueError(f"{val} has incorrect types, expected: (float, int)")


def _sanitize_weighting_rules(
    balance: Iterable[str], balance_weights: WeightingRulesDict
) -> InternalWeightingRulesDict:
    """Converts a `WeightingRulesDict` which supports a mix of single values/iterables,
    to a `InternalWeightingRulesDict` which only contains tuples."""
    if balance is None:
        raise ValueError("`balance_weights` cannot be used without `balance`.")

    balance_weights_fixed: WeightingRulesDict = {}
    # Return empty dict if no balance_weights are provided
    if balance_weights is None:
        return balance_weights_fixed

    # Convert individual values to tuples if needed and check the validity of the inputs.
    for k, v in balance_weights.items():
        # convert to tuple if needed
        k_new = _to_tuple(k)
        if isinstance(v, float):
            v_new = (v, 0)
        elif len(v) == 2 and isinstance(v[0], float) and isinstance(v[1], int):
            v_new = v
        else:
            raise ValueError(
                f"Value {v} is not a valid weighting value. Expected `priority` or `(weight, priority)`."
            )

        # raise error if not valid
        _raise_weighting_rule_key(k_new, balance)
        _raise_weighting_rule_value(v_new)

        # create rule
        balance_weights_fixed[k_new] = v_new
    return balance_weights_fixed


class TaskBalancingSpecifications:
    def __init__(
        self,
        keys: Iterable[str],
        weighting_rules: WeightingRulesDict = None,
    ):
        """Describe how to balance the content of batches in a task.

        Parameters
        ----------
        keys : Iterable[str]
            list of ProtocolFile keys that will be used to balance the batches.
            e.g. ['database', 'channel', 'speaker']
        weighting_rules : WeightingRulesDict, optional
            Rules to define the sampling weight of combinations of keys.
            This dictionary can be created dynamically using `add_weighting_rule`.
            - Keys are tuples indicating a "condition": a matching combinations of keys. `None` means any value.
            - Values are either a float (weight) or a tuple (weight, priority).
            The weight controls the weighted sampling (higher = sampled more often).
            The priority controls the order in which rules are checked (higher = more priority).
            Cases not covered by rules are assigned a weight of 1.0. Default priority is 0.
        See examples below for more details.

        Example 1
        ---------
        >>> from pyannote.audio.utils import TaskBalancingSpecifications
        >>> task_balance = TaskBalancingSpecifications(
        ...     keys=['database'],
        ...     weighting_rules={('AMI',): 3.0}
        ... )
        >>> # weights AMI files 3 times more than others

        Example 2
        ---------
        >>> from pyannote.audio.utils import TaskBalancingSpecifications
        >>> task_balance = TaskBalancingSpecifications(
        ...     keys=['database', 'domain', 'channel'],
        ...     weighting_rules={
        ...         ('DIHARD',): 3.0,
        ...         (None, 'audiobooks', 'CH01'): 0.1,
        ...         ('DIHARD', 'audiobooks', None): (10.0, 2),
        ...  })
        >>> # DIHARD files are sampled with a weight of 3.
        >>> # audiobooks files from channel CH01 are sampled with a weight of 0.1 (any database)
        >>> # DIHARD audiobooks files (any channel) are sampled with a weight of 10.0, this takes priority over the previous rule
        >>> # (the last rule's priority is set to 2)
        """
        self._keys: Iterable[str] = keys
        self._weight_rules: InternalWeightingRulesDict = {}

        self.set_weighting_rules(weighting_rules)

    def set_weighting_rules(
        self,
        weighting_rules: WeightingRulesDict,
    ) -> None:
        """Set the weighting rules for the task balance (discards previous rules).

        Parameters
        ----------
        weighting_rules : WeightingRulesDict
            New weighting rules. See class docstring for more details.
        """
        self._weight_rules = _sanitize_weighting_rules(self.keys, weighting_rules)

    def add_weighting_rule(
        self,
        condition: ConditionValue | tuple[ConditionValue],
        weighting: float,
        priority: int = 0,
    ):
        """Add one weighting rule.

        Parameters
        ----------
        condition : ConditionValue | tuple[ConditionValue]
            Balance values needed to apply the weighting rule.
        weighting : float
            Weighting rule (relative sampling weight when the condition is met).
        priority : int, optional
            Rule priority. The higher it is, the more priority is has over other conditions, by default 0
        """
        key = _to_tuple(condition)
        value = (weighting, priority)

        _raise_weighting_rule_key(key, self.keys)
        _raise_weighting_rule_value(value)

        self._weight_rules[key] = value

    def remove_weighting_rule(self, condition: tuple[str]):
        """Remove one rule.

        Parameters
        ----------
        condition : tuple[str]
            Condition of the rule to be removed.
        """
        del self._weight_rules[tuple]

    def compute_weights(self, tuples: list[tuple[ConditionValue, ...]]) -> list[float]:
        """Returns the relative weights of multiple conditions.

        Parameters
        ----------
        tuples : list[tuple[ConditionValue]]
            List of conditions, e.g. [('DIHARD', 0), ('AMI', 1), ('DIHARD', 1)]

        Returns
        -------
        list[float]
            List of weights, one per condition.
            e.g. [1.0, 3.0, 1.0]
        """
        subchunks_weights: list[float] = []
        for needed_tuple in tuples:
            if len(needed_tuple) != len(self.keys):
                raise ValueError("tuples must have the same size as TaskBalance.keys")

            matching_weight = 1.0  # default weight
            matching_best = 0
            for weight_tuple, (weight, priority) in self._weight_rules.items():
                p = _get_tuples_matching_power(weight_tuple, needed_tuple)
                p *= 1 + priority * len(self.keys)
                if p > matching_best:
                    matching_weight = weight
                    matching_best = p
                elif p == matching_best and p != 0:
                    raise ValueError("Ambiguity! Two tuples match with same priority.")

            subchunks_weights.append(matching_weight)
        return subchunks_weights

    def compute_cumweights(
        self, tuples: list[tuple[ConditionValue, ...]]
    ) -> list[float]:
        """Returns the result of `compute_weights` with a cumulative sum, for easier usage with random.choices.

        Parameters
        ----------
        tuples : list[tuple[ConditionValue]]
            List of conditions, e.g. [('DIHARD', 0), ('AMI', 1), ('DIHARD', 1)]

        Returns
        -------
        list[float]
            List of cumulative weights, one per condition.
            e.g. [1.0, 4.0, 5.0] (obtained from non cumulative weights [1.0, 3.0, 1.0])
        """
        subchunks_weights = self.compute_weights(tuples)
        return list(itertools.accumulate(subchunks_weights))

    @property
    def keys(self):
        return self._keys
