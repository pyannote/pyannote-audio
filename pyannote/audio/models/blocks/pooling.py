# MIT License
#
# Copyright (c) 2020 CNRS
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
        sequences : (batch, channel, frames) or (batch, speakers, channel, frame) 
            torch.Tensor Sequences.
        weights : (batch, frames) or (batch, speakers, frames) 
            torch.Tensor, optional
            When provided, compute weighted mean and standard deviation.

        Returns
        -------
        output : (batch, 2 * channel) or (batch, speakers, 2 * channel) 
            torch.Tensor
            Concatenation of mean and (unbiased) standard deviation, eventually
            for each speaker if 4D sequences tensor is provided in arguments.
        """
        
        if len(sequences.shape) == 3:
            frames_dim = 2
        else:
            frames_dim = 3

        if weights is None:
            mean = sequences.mean(dim=frames_dim)
            std = sequences.std(dim=frames_dim, unbiased=True)

        else:
            # Unsqueeze before frames dimension:
            weights = weights.unsqueeze(dim=frames_dim - 1)
            # (batch, 1, frames) or (batch, speakers, 1, frames)

            num_frames = sequences.shape[frames_dim]
            num_weights = weights.shape[frames_dim]
            if num_frames != num_weights:
                warnings.warn(
                    f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
                )
                weights = F.interpolate(
                    weights, size=num_frames, mode="linear", align_corners=False
                )

            v1 = weights.sum(dim=frames_dim)
            mean = torch.sum(sequences * weights, dim=frames_dim) / v1

            dx2 = torch.square(sequences - mean.unsqueeze(frames_dim))
            v2 = torch.square(weights).sum(dim=frames_dim)

            var = torch.sum(dx2 * weights, dim=frames_dim) / (v1 - v2 / v1)
            std = torch.sqrt(var)

        return torch.cat([mean, std], dim=frames_dim - 1)
