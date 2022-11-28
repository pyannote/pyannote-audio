import torch
import random

import itertools
from pyannote.audio.core.task import Problem


def get_multilabel_sample(
    batch_size, num_frames, max_num_speakers, max_simult_speakers=0, device=None
):
    """Util method to get a multilabel speaker activity tensor of shape (B,F,S) with at most max_simult_speakers at each frame."""
    if max_simult_speakers < 1:
        max_simult_speakers = max_num_speakers

    sample = torch.randint(
        0, 2, (batch_size, num_frames, max_num_speakers), device=device
    )
    if max_simult_speakers < max_num_speakers:
        s2, idx = torch.sort(sample, descending=True)
        s2[:, :, max_simult_speakers:] = 0  # remove superfluous speakers
        sample = torch.zeros_like(sample).scatter(2, idx, s2)
    return sample


def test_multi_to_mono_simple():
    sample = torch.zeros(1, 1, 3)
    active_speakers = torch.tensor([0, 2])
    sample[0, 0, active_speakers] = 1

    r = Problem.multilabel_to_powerset(
        sample, max_num_speakers=3, max_simult_speakers=2
    )

    # checks the right class is predicted
    assert torch.argmax(r, dim=-1).flatten().item() == 5
    # checks there is the right amount of classes
    assert r.shape[2] == 7


def test_multi_to_mono_class_count():
    MAX_SPEAKERS_TO_TEST = 6
    for max_speakers in range(1, MAX_SPEAKERS_TO_TEST):
        for max_simult_speakers in range(1, max_speakers):
            sample = torch.zeros(1, 1, max_speakers)
            sample_mono = Problem.multilabel_to_powerset(
                sample, max_speakers, max_simult_speakers
            )

            assert (
                Problem.get_powerset_class_count(max_speakers, max_simult_speakers)
                == sample_mono.shape[-1]
            )


def test_multi_to_mono_to_multi():
    BATCH_SIZE = 128
    NUM_FRAMES = 200
    MAX_SPEAKERS = 6
    MAX_SIMULT_SPEAKERS = 3
    sample = get_multilabel_sample(
        BATCH_SIZE, NUM_FRAMES, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS
    )
    sample_mono = Problem.multilabel_to_powerset(
        sample, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS
    )
    sample_multi = Problem.powerset_to_multilabel(
        sample_mono, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS
    )
    assert sample.shape == sample_multi.shape
    assert torch.all(sample.flatten() == sample_multi.flatten())


def test_multi_to_mono_too_many_simult():
    BATCH_SIZE = 128
    NUM_FRAMES = 200
    MAX_SPEAKERS = 6
    MAX_SIMULT_SPEAKERS = 3
    MAX_SIMULT_SPEAKERS_IN_SAMPLE = 5
    sample = get_multilabel_sample(
        BATCH_SIZE, NUM_FRAMES, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS_IN_SAMPLE
    )
    sample_mono = Problem.multilabel_to_powerset(
        sample, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS
    )
    sample_multi = Problem.powerset_to_multilabel(
        sample_mono, MAX_SPEAKERS, MAX_SIMULT_SPEAKERS
    )
    assert sample.shape == sample_multi.shape
    error_frames = torch.sum(sample != sample_multi, dim=-1) > 0
    too_many_simult_frames = torch.sum(sample, dim=-1) > MAX_SIMULT_SPEAKERS
    # check the errors are on the frames with too many speakers
    assert torch.all(error_frames == too_many_simult_frames)


def test_multi_to_mono_permutation():
    MAX_SPEAKERS = 6

    # tests that the mono-version of every permutations and their inverses
    # are still permutations and their inverses
    # (unless our function makes errors on everything but pairs of permutations and their inverses, unlikely)

    for max_speakers in range(1, MAX_SPEAKERS):
        for max_simult_speakers in range(1, max_speakers + 1):
            sample_t = torch.arange(
                0, Problem.get_powerset_class_count(max_speakers, max_simult_speakers)
            )

            for p_tuple in itertools.permutations([i for i in range(max_speakers)]):
                p = torch.tensor(p_tuple)
                p_inv = torch.argsort(p)
                p_mono = Problem.get_powerset_permutation(
                    p, max_speakers, max_simult_speakers
                )
                p_inv_mono = Problem.get_powerset_permutation(
                    p_inv, max_speakers, max_simult_speakers
                )

                assert torch.all(sample_t == sample_t[p_mono][p_inv_mono])
