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


from functools import partial
from typing import Callable, Dict, Set, Text

import torch.nn as nn


def probe(trunk: nn.Module, branches: Dict[Text, Text]) -> Callable:
    """Add probing branches to a trunk module

    Parameters
    ----------
    trunk : nn.Module
        Multi-layer trunk.
    branches : {branch_name: layer_name} dict or [layer_name] list
        Indicate where to plug a probing branch.

    Returns
    -------
    revert : Callable
        Callable that, when called, removes probing branches.

    Usage
    -----

    Define a trunk made out of three consecutive layers

    >>> import torch.nn as nn
    >>> class Trunk(nn.Module):
    ...
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.layer1 = nn.Linear(1, 2)
    ...         self.layer2 = nn.Linear(2, 3)
    ...         self.layer3 = nn.Linear(3, 4)
    ...
    ...     def forward(self, x):
    ...         return self.layer3(self.layer2(self.layer1(x)))

    >>> trunk = Trunk()
    >>> x = torch.tensor((0.,))
    >>> trunk(x)
    # tensor([ 0.4548, -0.1814,  0.9494,  1.0445], grad_fn=<AddBackward0>)

    Add two probing branches:
    - first one is called "probe1" and probes the output of "layer1"
    - second one is called "probe2" and probes the output of "layer3"

    >>> revert = probe(trunk, {"probe1": "layer1", "probe2": "layer3"})
    >>> trunk(x)
    # {'probe1': tensor([ 0.5854, -0.9685], grad_fn=<AddBackward0>),
    #  'probe2': tensor([ 0.4548, -0.1814,  0.9494,  1.0445], grad_fn=<AddBackward0>)}

    Use callback returned by `probe` to revert its effect

    >>> revert()
    >>> trunk(x)
    # tensor([ 0.4548, -0.1814,  0.9494,  1.0445], grad_fn=<AddBackward0>)

    For convenience, one can also define probes as a list of layers:

    >>> revert = probe(trunk, ['layer1', 'layer3'])
    >>> trunk(x)
    # {'layer1': tensor([ 0.5854, -0.9685], grad_fn=<AddBackward0>),
    #  'layer3': tensor([ 0.4548, -0.1814,  0.9494,  1.0445], grad_fn=<AddBackward0>)}
    """

    def remove():
        del trunk.__probe
        for handle in trunk.__probe_handles:
            handle.remove()
        del trunk.__probe_handles

    if hasattr(trunk, "__probe"):
        remove()

    trunk.__probe_handles = []

    def __probe_init(module, input):
        trunk.__probe = dict()

    handle = trunk.register_forward_pre_hook(__probe_init)
    trunk.__probe_handles.append(handle)

    def __probe_append(branch_name, module, input, output):
        trunk.__probe[branch_name] = output

    if not isinstance(branches, dict):
        branches = {b: b for b in branches}

    sehcnarb: Dict[Text, Set] = dict()
    for branch_name, layer_name in branches.items():
        if layer_name not in sehcnarb:
            sehcnarb[layer_name] = set()
        sehcnarb[layer_name].add(branch_name)

    for layer_name, layer in trunk.named_modules():
        if layer_name not in sehcnarb:
            continue
        for branch_name in sehcnarb[layer_name]:
            handle = layer.register_forward_hook(partial(__probe_append, branch_name))
            trunk.__probe_handles.append(handle)

    def __probe_return(module, input, output):
        return trunk.__probe

    handle = trunk.register_forward_hook(__probe_return)
    trunk.__probe_handles.append(handle)

    return remove
