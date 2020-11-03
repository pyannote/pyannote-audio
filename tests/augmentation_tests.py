import pytest
import torch
from torch.nn import Module

from pyannote.audio.augmentation.registry import (
    register_augmentation,
    unregister_augmentation,
)


class RandomAugmentation(Module):
    "Fake Noise tranform"

    def forward(self, waveforms):
        if not self.training:
            return waveforms
        return torch.ones_like(waveforms)


def test_can_register_augmentation():
    for when in ["output", "input"]:
        net = Module()
        aug = RandomAugmentation()
        register_augmentation(aug, net, when=when)
        assert hasattr(net, "__augmentation")
        assert net.__augmentation[when] == aug


def test_can_unregistrer_augmentation():
    net = Module()
    register_augmentation(RandomAugmentation(), net, when="output")
    unregister_augmentation(net, when="output")
    assert not hasattr(net.__augmentation, "output")


def test_fail_unregister_augmentation():
    with pytest.raises(ValueError):
        unregister_augmentation(RandomAugmentation(), when="output")
