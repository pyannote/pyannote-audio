# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StatsPool(nn.Module):
    """Statistics pooling

    Compute temporal mean and (unbiased) standard deviation
    and returns their concatenation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    """

    def _pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Helper function to compute statistics pooling

        Assumes that weights are already interpolated to match the number of frames
        in sequences and that they encode the activation of only one speaker.

        Parameters
        ----------
        sequences : (batch, features, frames) torch.Tensor
            Sequences of features.
        weights : (batch, frames) torch.Tensor
            (Already interpolated) weights.

        Returns
        -------
        output : (batch, 2 * features) torch.Tensor
            Concatenation of mean and (unbiased) standard deviation.
        """

        weights = weights.unsqueeze(dim=1)
        # (batch, 1, frames)

        v1 = weights.sum(dim=2) + 1e-8
        mean = torch.sum(sequences * weights, dim=2) / v1

        dx2 = torch.square(sequences - mean.unsqueeze(2))
        v2 = torch.square(weights).sum(dim=2)

        var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1 + 1e-8)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)

    def forward(
        self, sequences: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        sequences : (batch, features, frames) torch.Tensor
            Sequences of features.
        weights : (batch, frames) or (batch, speakers, frames) torch.Tensor, optional
            Compute weighted mean and standard deviation, using provided `weights`.

        Note
        ----
        `sequences` and `weights` might use a different number of frames, in which case `weights`
        are interpolated linearly to reach the number of frames in `sequences`.

        Returns
        -------
        output : (batch, 2 * features) or (batch, speakers, 2 * features) torch.Tensor
            Concatenation of mean and (unbiased) standard deviation. When `weights` are
            provided with the `speakers` dimension, `output` is computed for each speaker
            separately and returned as (batch, speakers, 2 * channel)-shaped tensor.
        """

        if weights is None:
            mean = sequences.mean(dim=-1)
            std = sequences.std(dim=-1, correction=1)
            return torch.cat([mean, std], dim=-1)

        if weights.dim() == 2:
            has_speaker_dimension = False
            weights = weights.unsqueeze(dim=1)
            # (batch, frames) -> (batch, 1, frames)
        else:
            has_speaker_dimension = True

        # interpolate weights if needed
        _, _, num_frames = sequences.shape
        _, _, num_weights = weights.shape
        if num_frames != num_weights:
            warnings.warn(
                f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
            )
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        output = rearrange(
            torch.vmap(self._pool, in_dims=(None, 1))(sequences, weights),
            "speakers batch features -> batch speakers features",
        )

        if not has_speaker_dimension:
            return output.squeeze(dim=1)

        return output
