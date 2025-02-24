import pytest

from pyannote.audio.utils.balance import TaskBalancingSpecifications


def test_task_balancing_specifications_simple():
    rules = {
        "DB1": 2.0,
        "DB2": 4.0,
    }
    specs1 = TaskBalancingSpecifications(
        ["database"],
        weighting_rules=rules,
    )
    tuples = [("DB1",), ("DB2",), ("DB3",)]
    assert specs1.compute_weights(tuples) == [2.0, 4.0, 1.0]

    specs2 = TaskBalancingSpecifications(
        ["database"],
    )
    for tuple, weight in rules.items():
        specs2.add_weighting_rule(tuple, weight)

    assert specs1.compute_weights(tuples) == specs2.compute_weights(tuples)


def test_task_balancing_specifications_advanced():
    rules = {
        ("DB1"): 2.0,
        ("DB2", "train", None): 4.0,
        (None, "train", "SPK1"): (8.0, 2),
    }
    specs1 = TaskBalancingSpecifications(
        ["database", "subset", "speaker"],
        weighting_rules=rules,
    )
    tuples = [
        ("DB1", "train", "SPK2"),
        ("DB2", "train", "SPK1"),
        ("DB2", "train", "SPK2"),
        ("DB3", "train", "SPK2"),
    ]
    assert specs1.compute_weights(tuples) == [2.0, 8.0, 4.0, 1.0]


def test_task_balancing_specifications_ambiguous():
    rules1 = {
        ("DB1", "train", None): 4.0,
        (None, "train", "SPK1"): 8.0,
    }
    specs1 = TaskBalancingSpecifications(
        ["database", "subset", "speaker"],
        weighting_rules=rules1,
    )
    with pytest.raises(ValueError):
        specs1.compute_weights([("DB1", "train", "SPK1")])

    # now with priority there shouldn't be any ambiguity left
    rules2 = {
        ("DB1", "train", None): (4.0, 2),
        (None, "train", "SPK1"): 8.0,
    }
    specs2 = TaskBalancingSpecifications(
        ["database", "subset", "speaker"],
        weighting_rules=rules2,
    )
    assert specs2.compute_weights([("DB1", "train", "SPK1")]) == [4.0]
