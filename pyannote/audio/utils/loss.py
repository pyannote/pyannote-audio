# MIT License
#
# Copyright (c) 2020-2021 CNRS
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


"""Frame-weighted versions of common loss functions"""

import torch
import torch.nn.functional as F


def interpolate(target: torch.Tensor, weight: torch.Tensor = None):
    """Interpolate weight to match target frame resolution

    Parameters
    ----------
    target : torch.Tensor
        Target with shape (batch_size, num_frames) or (batch_size, num_frames, num_classes)
    weight : torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames_weight, 1).

    Returns
    -------
    weight : torch.Tensor
        Interpolated frame weight with shape (batch_size, num_frames, 1).
    """

    num_frames = target.shape[1]
    if weight is not None and weight.shape[1] != num_frames:
        weight = F.interpolate(
            weight.transpose(1, 2),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return weight


def binary_cross_entropy(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
) -> torch.Tensor:
    """Frame-weighted binary cross entropy

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames) for binary or multi-class classification,
        or (batch_size, num_frames, num_classes) for multi-label classification.
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    # reshape target to (batch_size, num_frames, num_classes) even if num_classes is 1
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    if weight is None:
        return F.binary_cross_entropy(prediction, target.float())

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight)

        return F.binary_cross_entropy(
            prediction, target.float(), weight=weight.expand(target.shape)
        )


def mse_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
) -> torch.Tensor:
    """Frame-weighted mean-squared error loss

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames) for binary or multi-class classification,
        or (batch_size, num_frames, num_classes) for multi-label classification.
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    # reshape target to (batch_size, num_frames, num_classes) even if num_classes is 1
    if len(target.shape) == 2:
        target = target.unsqueeze(dim=2)

    losses = F.mse_loss(prediction, target.float(), reduction="none")
    # (batch_size, num_frames, num_classes)

    if weight is None:
        return torch.mean(losses)

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight).expand(losses.shape)
        # (batch_size, num_frames, num_classes)

        return torch.sum(losses * weight) / torch.sum(weight)


def nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor = None,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    """Frame-weighted negative log-likelihood loss

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction with shape (batch_size, num_frames, num_classes).
    target : torch.Tensor
        Target with shape (batch_size, num_frames)
    class_weight : (num_classes, ) torch.Tensor, optional
        Class weight with shape (num_classes,  )
    weight : (batch_size, num_frames, 1) torch.Tensor, optional
        Frame weight with shape (batch_size, num_frames, 1).

    Returns
    -------
    loss : torch.Tensor
    """

    num_classes = prediction.shape[2]

    losses = F.nll_loss(
        prediction.view(-1, num_classes),
        # (batch_size x num_frames, num_classes)
        target.view(-1),
        # (batch_size x num_frames, )
        weight=class_weight,
        # (num_classes, )
        reduction="none",
    ).view(target.shape)
    # (batch_size, num_frames)

    if weight is None:
        return torch.mean(losses)

    else:
        # interpolate weight
        weight = interpolate(target, weight=weight).squeeze(dim=2)
        # (batch_size, num_frames)

        return torch.sum(losses * weight) / torch.sum(weight)
