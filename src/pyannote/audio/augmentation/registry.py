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

from functools import singledispatch

import torch
import torch.nn as nn
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.model import Model


def register_augmentation(
    augmentation: nn.Module,
    module: nn.Module,
    when: str = "input",
):
    """Register augmentation

    Parameters
    ----------
    augmentation : nn.Module
        Augmentation module.
    module : nn.Module
        Module whose input or output should be augmented.
    when : {'input', 'output'}
        Whether to apply augmentation on the input or the output.
        Defaults to 'input'.

    Usage
    -----

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectogram = Spectrogram()
            self.other_layers = nn.Identity()

        def forward(self, waveforms):
            spectrogram = self.spectrogram(waveforms)
            return self.other_layers = other_layers

    net = Net()

    class AddNoise(nn.Module):
        def forward(self, waveforms):
            if not self.training:
                return waveforms

            augmented_waveforms = ...
            return augmented_waveforms

    # AddNoise will be automatically applied to `net` input
    register_augmentation(AddNoise(), net, when='input')

    class SpecAugment(nn.Module):
        def forward(self, spectrograms):
            if not self.training:
                return spectrograms

            augmented_spectrograms = ...
            return augmented_spectrograms

    # SpecAugment will be automatically applied to `net.spectrogram` output
    register_augmentation(SpecAugment(), net.spectrogram, when='output')

    # deactivate augmentations
    net.eval()  # or net.train(mode=False)

    # reactivate augmentations
    net.train()

    # unregister "AddNoise" augmentation
    unregister_augmentation(net, when='input')

    """

    wrapped_augmentation = wrap_augmentation(augmentation, module, when=when)

    if not hasattr(module, "__augmentation"):
        module.__augmentation = nn.ModuleDict()
        module.__augmentation_handle = dict()

    # unregister any augmentation that might already exist
    if when in module.__augmentation:
        unregister_augmentation(module, when=when)

    module.__augmentation[when] = wrapped_augmentation

    if when == "input":

        def input_hook(augmented_module, input):
            return wrapped_augmentation(*input)

        handle = module.register_forward_pre_hook(input_hook)

    elif when == "output":

        def output_hook(augmented_module, input, output):
            return wrapped_augmentation(output)

        handle = module.register_forward_hook(output_hook)

    module.__augmentation_handle[when] = handle


def unregister_augmentation(module: nn.Module, when: str = "input"):
    """Unregister augmentation

    Parameters
    ----------
    module : nn.Module
        Module whose augmentation should be removed.
    when : {'input', 'output'}
        Whether to remove augmentation of the input or the output.
        Defaults to 'input'.

    Raises
    ------
    ValueError if module has no corresponding registered augmentation.
    """

    if (not hasattr(module, "__augmentation")) or (when not in module.__augmentation):
        raise ValueError(f"Module has no registered {when} augmentation.")

    del module.__augmentation[when]

    # unregister forward hook using previously stored handle
    handle = module.__augmentation_handle.pop(when)
    handle.remove()


@singledispatch
def wrap_augmentation(augmentation, model: Model, when: str = "input"):
    return augmentation


#  =============================================================================
#  Support for torch-audiomentations waveform transforms
#  =============================================================================


class TorchAudiomentationsWaveformTransformWrapper(nn.Module):
    def __init__(
        self, augmentation: BaseWaveformTransform, model: Model, when: str = "input"
    ):
        super().__init__()

        self.augmentation = augmentation

        if not isinstance(model, Model):
            raise TypeError(
                f"torch-audiomentations waveform transforms can only be applied to `pyannote.audio.Model` instances: "
                f"you tried with a {model.__class__.__name__} instance."
            )
        if when != "input":
            raise ValueError(
                f"torch-audiomentations waveform transforms can only be applied to the model input: "
                f"you tried with the {when}."
            )

        self.sample_rate_ = model.audio.sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.augmentation(
            samples=waveforms, sample_rate=self.sample_rate_
        ).samples


@wrap_augmentation.register
def _(augmentation: BaseWaveformTransform, model: Model, when: str = "input"):
    return TorchAudiomentationsWaveformTransformWrapper(augmentation, model, when=when)


# TODO: add support for future torch-audiomentations Compose transforms
# See https://github.com/asteroid-team/torch-audiomentations/issues/23
