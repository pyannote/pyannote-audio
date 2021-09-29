# MIT License
#
# Copyright (c) 2021 CNRS
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


from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR as _CyclicLR


def CyclicLR(
    optimizer: Optimizer,
    min_lr: float = 1e-8,
    max_lr: float = 1e-3,
    mode: str = "triangular2",
    patience: int = 50,
    num_batches_per_epoch: int = None,
    **kwargs,
):
    """Wrapper around CyclicLR learning rate scheduler

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer
    min_lr : float, optional
        Defaults to 1e-8.
    max_lr : float, optional
        Defaults to 1e-3
    patience : int, optional
        Number of epochs per cycle. Defaults to 50.
    num_batches_per_epoch : int, optional
        Number of batches per epoch.
    mode : {"triangular", "triangular2"}, optional
        Defaults to "triangular2".
    """

    step_size_up = int(0.5 * patience * num_batches_per_epoch)

    return {
        "scheduler": _CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode,
            cycle_momentum=False,
        ),
        "interval": "step",
    }
