import itertools
from typing import Any, Dict, Iterable, List, Text, Tuple, Union


def _get_tuples_matching_power(tuple1: Tuple, tuple2: Tuple) -> int:
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
        elif tuple1[i] == None:
            continue
        else:
            return 0
    return matching_elements

def _to_weighting_rule_tuple(val: Union[Text, Iterable[Text], Tuple[Text]]) -> Tuple[Text]:
    if (
        not isinstance(val, str)
        and not isinstance(val, tuple)
        and isinstance(val, Iterable)
    ):
        return tuple(val)
    if isinstance(val, str):
        return (val,)
    return val

def _to_weighting_rule_data(val: Union[float, Tuple[float, int]]) -> Tuple[float, int]:
    if isinstance(val, float):
        return (val, 0)
    return val

def _check_weighting_rule_tuple(val: Any, balance: List[Text]) -> None:
    if not isinstance(val, Tuple) or len(val) > len(balance):
        raise ValueError(f"{val} is not a matching tuple rule.")

def _check_weighting_rule_data(val: Any) -> None:
    if not isinstance(val, Tuple) or len(val) != 2:
        raise ValueError(f"{val} is not a tuple (weight, priority)")
    if not isinstance(val[0], float) or not isinstance(val[1], int):
        raise ValueError(f"{val} has incorrect types, expected: (float, int)")

def _get_fixed_balance_weights(balance, balance_weights):
    if balance is None:
        raise ValueError("`balance_weights` cannot be used without `balance`.")

    balance_weights_fixed = {}
    if balance_weights is None:
        return balance_weights_fixed

    for k, v in balance_weights.items():
        k_new = _to_weighting_rule_tuple(k)
        v_new = _to_weighting_rule_data(v)

        _check_weighting_rule_tuple(k_new, balance)
        _check_weighting_rule_data(v_new)
        
        balance_weights_fixed[k_new] = v_new
    return balance_weights_fixed


class TaskBalancingSpecifications:
    def __init__(
        self,
        keys: List[Text],
        weighting_rules: Dict[
            Union[Text, Tuple[Text]], Union[float, Tuple[float, int]]
        ] = None,
    ):
        """Describe how to balance the content of batches in a task.


        Parameters
        ----------
        keys : List[Text]
            List of ProtocolFile keys that will be used to balance the batches.
            e.g. ['database', 'channel', 'speaker']
        weighting_rules : Dict[ Union[Text, Tuple[Text]], Union[float, Tuple[float, int]] ], optional
            Rules to define the sampling weight of combinations of keys.
            This dictionary can be created dynamically using `add_weighting_rule`.
            Keys are tuple indicating matching combinations of keys, None means any value.
            Values are either a float or a tuple (float, int), where the float is the weight
            and the optional int is the rule priority (default is 1).
            Cases not covered by rules are assigned a weight of 1.0.
            See examples for more details.

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
        >>> # weights DIHARD files 3 times more than others
        >>> # weights audiobooks files from channel CH01 10 times less than others
        >>> # weights DIHARD audiobooks files 10 times more than others, this takes priority over the previous rule
        >>> # (the rule priority is set to 2)

        """
        self._keys = keys
        self._weight_rules = {}
        if weighting_rules is not None:
            self.set_weighting_rules(weighting_rules)

    def check_valid(self):
        for tuple in self._weight_rules.keys():
            if len(tuple) > len(self.keys):
                raise ValueError(
                    f"Tuple {tuple} is not valid, it should be of length {len(self.keys)}"
                )

    def set_weighting_rules(
        self,
        weighting_rules: Dict[
            Union[Text, Tuple[Text]], Union[float, Tuple[float, int]]
        ],
    ) -> None:
        self._weight_rules = _get_fixed_balance_weights(self.keys, weighting_rules)

    def add_weighting_rule(self, tuple: Union[Text,Tuple[Text]], weighting: float, priority=0):
        tuple = _to_weighting_rule_tuple(tuple)
        data = (weighting, priority)
        _check_weighting_rule_tuple(tuple, self.keys)
        _check_weighting_rule_data(data)
        self._weight_rules[tuple] = data

    def remove_weighting_rule(self, tuple: Tuple[Text]):
        del self._weight_rules[tuple]

    def compute_weights(self, tuples: List[Tuple]) -> List[float]:
        subchunks_weights: List[float] = []
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

    def compute_cumweights(self, tuples: List[Tuple]) -> List[float]:
        subchunks_weights = self.compute_weights(tuples)
        return list(itertools.accumulate(subchunks_weights))

    @property
    def keys(self):
        return self._keys
