from typing import List

import torch
import torchaudio
from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


class Reverb(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self, *args, sample_rate: int = 16000, params: List = [], **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.effects = [["reverb"] + params]

    def apply_transform(self, samples: Tensor, sample_rate: int):
        if samples.dim() == 2:
            samples = samples[None]
        return torch.stack([self.reverb(x) for x in samples])

    def reverb(self, waveform: Tensor):
        wave, sr = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, self.effects
        )
        return wave[0:1]
