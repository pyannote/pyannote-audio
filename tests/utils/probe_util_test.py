import torch
import torch.nn as nn

from pyannote.audio.utils.probe import probe


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 2)
        self.layer2 = nn.Linear(2, 3)
        self.layer3 = nn.Linear(3, 4)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


def test_probe_dict():
    trunk = Trunk()
    probe(trunk, {"probe1": "layer1"})
    out = trunk(
        torch.ones(
            1,
        )
    )
    assert isinstance(out, dict)
    assert len(out.keys()) == 1
    assert isinstance(out["probe1"], torch.Tensor)


def test_probe_output():
    trunk = Trunk()
    probe(trunk, {"probe1": "layer3"})
    out = trunk(
        torch.ones(
            1,
        )
    )
    out = out["probe1"]
    tout = trunk.layer3(
        trunk.layer2(
            trunk.layer1(
                torch.ones(
                    1,
                )
            )
        )
    )
    assert torch.equal(tout, out)


def test_probe_revert():
    trunk = Trunk()
    revert = probe(trunk, {"probe1": "layer3"})
    out = trunk(
        torch.ones(
            1,
        )
    )
    assert isinstance(out, dict)
    revert()
    out = trunk(
        torch.ones(
            1,
        )
    )
    assert isinstance(out, torch.Tensor)


def test_probe_array():
    trunk = Trunk()
    probe(trunk, ["layer3"])
    out = trunk(
        torch.ones(
            1,
        )
    )
    assert isinstance(out, dict)
