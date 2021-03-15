import numpy as np
import torch

from pyannote.audio.utils.permutation import permutate


def test_permutate_torch():

    num_frames, num_speakers = 10, 3

    actual_permutations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    batch_size = len(actual_permutations)

    y2 = torch.randn((num_frames, num_speakers))
    y1 = torch.zeros((batch_size, num_frames, num_speakers))

    for p, permutation in enumerate(actual_permutations):
        y1[p] = y2[:, permutation]

    permutated_y2, permutations = permutate(y1, y2)
    assert actual_permutations == permutations

    for p, permutation in enumerate(actual_permutations):
        np.testing.assert_allclose(permutated_y2[p], y2[:, permutation])


def test_permutate_numpy():

    num_frames, num_speakers = 10, 3

    actual_permutations = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    batch_size = len(actual_permutations)

    y2 = np.random.randn(num_frames, num_speakers)
    y1 = np.zeros((batch_size, num_frames, num_speakers))

    for p, permutation in enumerate(actual_permutations):
        y1[p] = y2[:, permutation]

    permutated_y2, permutations = permutate(y1, y2)
    assert actual_permutations == permutations

    for p, permutation in enumerate(actual_permutations):
        np.testing.assert_allclose(permutated_y2[p], y2[:, permutation])


def test_permutate_less_speakers():

    num_frames = 10

    actual_permutations = [
        (0, 1, None),
        (0, None, 1),
        (1, 0, None),
        (1, None, 0),
        (None, 0, 1),
        (None, 1, 0),
    ]
    batch_size = len(actual_permutations)

    y2 = np.random.randn(num_frames, 2)
    y1 = np.zeros((batch_size, num_frames, 3))

    for p, permutation in enumerate(actual_permutations):
        for i, j in enumerate(permutation):
            if j is not None:
                y1[p, :, i] = y2[:, j]

    permutated_y2, permutations = permutate(y1, y2)

    assert permutations == actual_permutations


def test_permutate_more_speakers():

    num_frames = 10

    actual_permutations = [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    batch_size = len(actual_permutations)

    y2 = np.random.randn(num_frames, 3)
    y1 = np.zeros((batch_size, num_frames, 2))

    for p, permutation in enumerate(actual_permutations):
        for i, j in enumerate(permutation):
            y1[p, :, i] = y2[:, j]

    permutated_y2, permutations = permutate(y1, y2)

    assert permutations == actual_permutations
    np.testing.assert_allclose(permutated_y2, y1)
