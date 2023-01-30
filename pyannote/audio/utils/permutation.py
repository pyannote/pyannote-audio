# MIT License
#
# Copyright (c) 2020-2022 CNRS
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


import math
from typing import Callable, List, Optional, Tuple
from functools import singledispatch, partial

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from pyannote.core import SlidingWindowFeature


@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None, return_cost: bool = False):
    """Find cost-minimizing permutation

    Parameters
    ----------
    y1 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_classes_1)
    y2 : np.ndarray or torch.Tensor
        (num_samples, num_classes_2) or (batch_size, num_samples, num_classes_2)
    cost_func : callable
        Takes two (num_samples, num_classes) sequences and returns (num_classes, ) pairwise cost.
        Defaults to computing mean squared error.
    return_cost : bool, optional
        Whether to return cost matrix. Defaults to False.

    Returns
    -------
    permutated_y2 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_classes_1)
    permutations : list of tuple
        List of permutations so that permutation[i] == j indicates that jth speaker of y2
        should be mapped to ith speaker of y1.  permutation[i] == None when none of y2 speakers
        is mapped to ith speaker of y1.
    cost : np.ndarray or torch.Tensor, optional
        (batch_size, num_classes_1, num_classes_2)
    """
    raise TypeError()


def mse_cost_func(Y, y, **kwargs):
    """Compute class-wise mean-squared error

    Parameters
    ----------
    Y, y : (num_frames, num_classes) torch.tensor

    Returns
    -------
    mse : (num_classes, ) torch.tensor
        Mean-squared error
    """
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


def mae_cost_func(Y, y, **kwargs):
    """Compute class-wise mean absolute difference error

    Parameters
    ----------
    Y, y: (num_frames, num_classes) torch.tensor

    Returns
    -------
    mae : (num_classes, ) torch.tensor
        Mean absolute difference error
    """
    return torch.mean(torch.abs(Y - y), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_classes_2 = y2.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if return_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
        # y1_ is (num_samples, num_classes_1)-shaped
        # y2_ is (num_samples, num_classes_2)-shaped
        with torch.no_grad():
            cost = torch.stack(
                [
                    cost_func(y2_, y1_[:, i : i + 1].expand(-1, num_classes_2))
                    for i in range(num_classes_1)
                ],
            )

        if num_classes_2 > num_classes_1:
            padded_cost = F.pad(
                cost,
                (0, 0, 0, num_classes_2 - num_classes_1),
                "constant",
                torch.max(cost) + 1,
            )
        else:
            padded_cost = cost

        permutation = [None] * num_classes_1
        for k1, k2 in zip(*linear_sum_assignment(padded_cost.cpu())):
            if k1 < num_classes_1:
                permutation[k1] = k2
                permutated_y2[b, :, k1] = y2_[:, k2]
        permutations.append(tuple(permutation))

        if return_cost:
            costs.append(cost)

    if return_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations


@permutate.register
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    output = permutate(
        torch.from_numpy(y1),
        torch.from_numpy(y2),
        cost_func=cost_func,
        return_cost=return_cost,
    )

    if return_cost:
        permutated_y2, permutations, costs = output
        return permutated_y2.numpy(), permutations, costs.numpy()

    permutated_y2, permutations = output
    return permutated_y2.numpy(), permutations


def build_permutation_graph(
    segmentations: SlidingWindowFeature,
    onset: float = 0.5,
    cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mae_cost_func,
) -> nx.Graph:
    """Build permutation graph

    Parameters
    ----------
    segmentations : (num_chunks, num_frames, local_num_speakers)-shaped SlidingWindowFeature
        Raw output of segmentation model.
    onset : float, optionan
        Threshold above which a speaker is considered active. Defaults to 0.5
    cost_func : callable
        Cost function used to find the optimal bijective mapping between speaker activations
        of two overlapping chunks. Expects two (num_frames, num_classes) torch.tensor as input
        and returns cost as a (num_classes, ) torch.tensor. Defaults to mae_cost_func.

    Returns
    -------
    permutation_graph : nx.Graph
        Nodes are (chunk_idx, speaker_idx) tuples.
        An edge between two nodes indicate that those are likely to be the same speaker
        (the lower the value of "cost" attribute, the more likely).
    """

    cost_func = partial(cost_func, onset=onset)

    chunks = segmentations.sliding_window
    num_chunks, num_frames, _ = segmentations.data.shape
    max_lookahead = math.floor(chunks.duration / chunks.step - 1)
    lookahead = 2 * (max_lookahead,)

    permutation_graph = nx.Graph()

    for C, (chunk, segmentation) in enumerate(segmentations):
        for c in range(max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)):

            if c == C:
                continue

            # extract common temporal support
            shift = round((C - c) * num_frames * chunks.step / chunks.duration)

            if shift < 0:
                shift = -shift
                this_segmentations = segmentation[shift:]
                that_segmentations = segmentations[c, : num_frames - shift]
            else:
                this_segmentations = segmentation[: num_frames - shift]
                that_segmentations = segmentations[c, shift:]

            # find the optimal one-to-one mapping
            _, (permutation,), (cost,) = permutate(
                this_segmentations[np.newaxis],
                that_segmentations,
                cost_func=cost_func,
                return_cost=True,
            )

            for this, that in enumerate(permutation):

                this_is_active = np.any(this_segmentations[:, this] > onset)
                that_is_active = np.any(that_segmentations[:, that] > onset)

                if this_is_active:
                    permutation_graph.add_node((C, this))

                if that_is_active:
                    permutation_graph.add_node((c, that))

                if this_is_active and that_is_active:
                    permutation_graph.add_edge(
                        (C, this), (c, that), cost=cost[this, that]
                    )

    return permutation_graph
