import numpy as np
import pytest
from scipy.cluster.hierarchy import is_valid_linkage

from pyannote.core.utils.hierarchy import pool


@pytest.mark.parametrize("pooling_func", ["average", "centroid"])
def test_pool_unconstrained(pooling_func):
    tree = pool(np.array([[0.0], [1.0], [10.0]]), pooling_func=pooling_func)
    assert is_valid_linkage(tree)
    np.testing.assert_array_equal(tree[:, 3], [2, 3])
    assert set(tree[0, :2]) == {0, 1}
    assert set(tree[1, :2]) == {2, 3}
    np.testing.assert_allclose(
        tree[:, 2], [1, 9.5 if pooling_func == "average" else 10]
    )


def test_pool_orders_larger_child_first_during_must_link_merges():
    tree = pool(
        np.array([[0.0], [1.0], [2.0], [20.0]]),
        must_link=[(0, 1), (1, 2)],
    )
    assert is_valid_linkage(tree)
    # The second merge combines cluster 4 (two points) with point 2.
    # The final merge combines cluster 5 (three points) with point 3.
    np.testing.assert_array_equal(tree[1:, :2], [[4, 2], [5, 3]])
    np.testing.assert_allclose(tree[:, 2:], [[0, 2], [0, 3], [19, 4]])


def test_pool_cannot_link_remains_separate_until_infinite_distance():
    tree = pool(np.array([[0.0], [1.0], [10.0]]), cannot_link=[(0, 1)])
    assert set(tree[0, :2]) == {1, 2}
    assert np.isinf(tree[-1, 2])
    assert tree[-1, 3] == 3


def test_pool_conflicting_constraints():
    with pytest.raises(ValueError, match="conflict"):
        pool(np.array([[0.0], [1.0]]), must_link=[(0, 1)], cannot_link=[(0, 1)])
