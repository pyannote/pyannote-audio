from functools import singledispatch
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None, returns_cost: bool = False):
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
    returns_cost : bool, optional
        Whether to return cost matrix. Defaults to False.

    Returns
    -------
    permutated_y2 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_classes_1)
    permutations : list of tuple
        List of permutations so that permutation[i] == j indicates that jth speaker of y2
        should be mapped to ith speaker of y1.
    cost : np.ndarray or torch.Tensor, optional
        (batch_size, num_classes_1, num_classes_2)
    """
    raise TypeError()


def mse_cost_func(Y, y):
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    returns_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes_1)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_classes_2 = y2.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if returns_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
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

        if returns_cost:
            costs.append(cost)

    if returns_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations


@permutate.register
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = None,
    returns_cost: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    output = permutate(
        torch.from_numpy(y1),
        torch.from_numpy(y2),
        cost_func=cost_func,
        returns_cost=returns_cost,
    )

    if returns_cost:
        permutated_y2, permutations, costs = output
        return permutated_y2.numpy(), permutations, costs.numpy()

    permutated_y2, permutations = output
    return permutated_y2.numpy(), permutations
