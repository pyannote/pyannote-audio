# MIT License
#
# Copyright (c) 2022- CNRS
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


from typing import Optional

import torch
from torch import Tensor
from torch_audiomentations import Mix


class MixSpeakerDiarization(Mix):
    """
    Create a new sample by mixing it with another random sample from the same batch

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    Parameters
    ----------
    min_snr_in_db : float, optional
        Defaults to 0.0
    max_snr_in_db : float, optional
        Defaults to 5.0
    max_num_speakers: int, optional
        Maximum number of speakers in mixtures.  Defaults to actual maximum number
        of speakers in each batch.
    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = True

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        max_num_speakers: int = None,
        output_type: str = "tensor",
    ):
        super().__init__(
            min_snr_in_db=min_snr_in_db,
            max_snr_in_db=max_snr_in_db,
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.max_num_speakers = max_num_speakers

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape
        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # count number of active speakers per sample
        num_speakers: torch.Tensor = torch.sum(torch.any(targets, dim=-2), dim=-1)
        max_num_speakers = self.max_num_speakers or torch.max(num_speakers)

        # randomize index of second sample, constrained by the fact that the
        # resulting mixture should have less than max_num_speakers
        self.transform_parameters["sample_idx"] = torch.arange(
            batch_size, dtype=torch.int64
        )
        for n in range(max_num_speakers + 1):

            # indices of samples with exactly n speakers
            samples_with_n_speakers = torch.where(num_speakers == n)[0]
            num_samples_with_n_speakers = len(samples_with_n_speakers)
            if num_samples_with_n_speakers == 0:
                continue

            # indices of candidate samples for mixing (i.e. samples that would)
            candidates = torch.where(num_speakers + n <= max_num_speakers)[0]
            num_candidates = len(candidates)
            if num_candidates == 0:
                continue

            # sample uniformly from candidate samples
            selected_candidates = candidates[
                torch.randint(
                    0,
                    num_candidates,
                    (num_samples_with_n_speakers,),
                    device=samples.device,
                )
            ]
            self.transform_parameters["sample_idx"][
                samples_with_n_speakers
            ] = selected_candidates
