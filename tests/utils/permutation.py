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
