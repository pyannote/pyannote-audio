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


from typing import Optional, Text

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau


def ReduceLROnPlateau(
    optimizer: Optimizer,
    monitor: Optional[Text] = None,
    direction: Optional[Text] = "min",
    min_lr: float = 1e-8,
    max_lr: float = 1e-3,
    factor: float = 0.5,
    patience: int = 50,
    **kwargs,
):
    """Wrapper around ReduceLROnPlateau learning rate scheduler

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer
    min_lr : float, optional
        Defaults to 1e-8.
    max_lr : float, optional
        Defaults to 1e-3
    factor : float, optional
        Defaults to 0.5
    patience : int, optional
        Wait that many epochs with no improvement before reducing the learning rate.
        Defaults to 50.
    monitor : str, optional
        Value to monitor
    direction : {"min", "max"}, optional
        "min" (resp. "max") means smaller (resp. larger) is better.
    """

    # initialize optimizer lr to max_lr
    for g in optimizer.param_groups:
        g["lr"] = max_lr

    return {
        "scheduler": _ReduceLROnPlateau(
            optimizer,
            mode=direction,
            factor=factor,
            patience=patience,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=min_lr,
            eps=1e-08,
            verbose=False,
        ),
        "interval": "epoch",
        "monitor": monitor,
        "strict": True,
    }
