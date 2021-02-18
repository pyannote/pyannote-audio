# MIT License
#
# Copyright (c) 2020-2021 CNRS
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

from typing import Mapping, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core.memory import ModelSummary

from pyannote.audio import Model


class GraduallyUnfreeze(Callback):
    """Gradually unfreeze layers

    1. Start training with all layers frozen, but those that depends on the task
       (i.e. those instantiated in model.build() and task.setup_loss_func()
    2. Train for a few epochs and unfreeze a few more layers
    3. Repeat

    Parameters
    ----------
    patience : int or dict, optional
        If `dict`, it should use the following convention:
            {epoch: list of names of layers to unfreeze at this epoch}
        For instance, {10: ["linear",], 15: ["lstm", "sincnet"]} will unfreeze
        "linear" at epoch #10 and "lstm" and "sincnet" at epoch #15.
        If `int`, unfreezes one more layer every `patience` epochs, starting from
        layers closer to the output up to layers closer to the input.

    Usage
    -----
    >>> callback = GraduallyUnfreeze(patience=10)
    >>> callback = GraduallyUnfreeze(patience={10: ["linear", ], 15: ["lstm", ]})
    >>> Trainer(callbacks=[callback]).fit(model)
    """

    def __init__(self, patience: Union[Mapping, int] = 1):
        super().__init__()
        self.patience = patience

    def on_fit_start(self, trainer: Trainer, model: Model):

        if isinstance(self.patience, int):
            self._schedule = {}
            summary = ModelSummary(model, mode="top")
            step = 1
            for layer, _ in reversed(summary.named_modules):
                if layer in model.task_dependent:
                    continue
                self._schedule[step * self.patience] = [layer]
                step += 1
                model.freeze_by_name(layer)

        elif isinstance(self.patience, Mapping):
            self._schedule = dict(self.patience)

    def on_train_epoch_start(self, trainer: Trainer, model: Model):
        for layer in self._schedule.get(trainer.current_epoch, list()):
            model.unfreeze_by_name(layer)
