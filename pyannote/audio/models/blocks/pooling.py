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


class StatsPool(nn.Module):
    """Statistics pooling

    Compute temporal mean and (unbiased) standard deviation
    and returns their concatenation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    """

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

        else:
            # fix the error setting the weights of previous layers to NaN during a backward step,
            # in the case where the weights for a speaker are all zero
            weights = weights + 1e-16
            # Unsqueeze before frames dimension to match with waveforms dimensions
            weight_dims = len(weights.shape)
            weights = weights.unsqueeze(dim=-2)
            if weight_dims == 3:
                sequences = sequences.unsqueeze(dim=1)

            # (batch, 1, weights) or (batch, speakers, weights)
            num_frames = sequences.shape[-1]
            num_weights = weights.shape[-1]
            if num_frames != num_weights:
                warnings.warn(
                    f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
                )
                weights = F.interpolate(
                    weights, size=num_frames, mode="linear", align_corners=False
                )

            # add an epsilon value to avoid division by zero when the sum of the weights is 0
            v1 = weights.sum(dim=-1) + 1e-8
            mean = torch.sum(sequences * weights, dim=-1) / v1

            dx2 = torch.square(sequences - mean.unsqueeze(-1))
            v2 = torch.square(weights).sum(dim=-1)

            var = torch.sum(dx2 * weights, dim=-1) / ((v1 - v2 / v1) + 1e-8)
            std = torch.sqrt(var)

        return torch.cat([mean, std], dim=-1)
