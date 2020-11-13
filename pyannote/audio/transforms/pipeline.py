from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from torch.nn import Sequential
from torch_audiomentations.augmentations.background_noise import ApplyBackgroundNoise
from torchaudio.transforms import MFCC, FrequencyMasking, Resample, TimeMasking

from pyannote.audio.transforms.transforms import Reverb

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from pyannote.audio.core.task import Task


class TransformsPipe(Sequential):
    """
    A Sequential with transforms which are applied to data such as a Spectrogram
    Transforms that are probabalistic, can be returned for validation
    and testing with the `validate_pipe` command.
    """

    def __init__(self, *args):
        args = map(lambda a: TransformsPipe(*a) if isinstance(a, list) else a, args)
        super().__init__(*args)

    @property
    def validate_pipe(self) -> TransformsPipe:
        return TransformsPipe([a for a in self if not hasattr(a, "p") or a.p < 1])

    def forward(self, x: Tensor) -> Tensor:
        for module in self:
            x = module(x)
        return x


def default_transforms(task: Task, bg_noise: Union[str, Path] = None) -> TransformsPipe:
    """Basic Augmentations for waveform to spectrogram


    Parameters
    ---------
    task : Task
        The task that need transformations to be applied to
    bg_noise: Path, str
        Where to find background noise data

    Returns
    -------

    pipe
    """
    sr = task.audio.sample_rate
    augs = []
    reverb = Reverb(sample_rate=sr)
    # Test and validation augmentations will be applied without the
    # random transformations
    reverb._random = True
    augs += [Resample(sr), reverb]

    # Setup Background Noise
    if bg_noise:
        augs.append(ApplyBackgroundNoise(bg_noise))

    kwargs = _override_bad_defaults({})
    mfcc = MFCC(sr, melkwargs=kwargs)
    spec = mfcc(task.example_input_array)
    time, freq = spec.shape[-2:]

    # These Spec Augments need to be replaced with the ones
    # that stabalised training in the paper.
    timemasking = TimeMasking(math.floor(time) * 0.2, True)
    timemasking._random = True
    freqmasking = FrequencyMasking(math.floor(freq * 0.2), True)
    freqmasking._random = True
    augs += [MFCC(sr), timemasking, freqmasking]
    return TransformsPipe(augs)


def _override_bad_defaults(kwargs):
    "Fix bad default for spectrograms"

    if "n_fft" not in kwargs or kwargs["n_fft"] is None:
        kwargs["n_fft"] = 1024
    if "win_length" not in kwargs or kwargs["win_length"] is None:
        kwargs["win_length"] = kwargs["n_fft"]
    if "hop_length" not in kwargs or kwargs["hop_length"] is None:
        kwargs["hop_length"] = int(kwargs["win_length"] / 2)
    return kwargs


__all__ = default_transforms.__name__, TransformsPipe.__name__
