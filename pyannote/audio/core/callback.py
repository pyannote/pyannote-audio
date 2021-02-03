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


from pytorch_lightning import Callback, Trainer

from pyannote.audio import Model


class GraduallyUnfreeze(Callback):
    """Gradually unfreeze layers

    1. Freezes all layers but those that depends on the task.
    2. Waits for a few training epochs to pass.
    3. Unfreezes all layers.

    Parameters
    ----------
    patience : int, optional
        Wait for that many epochs before unfreezing all layers.
        Defaults to 1.
    """

    def __init__(self, patience: int = 1):
        super().__init__()
        self.patience = patience

    def on_fit_start(self, trainer: Trainer, model: Model):
        self._task_independent_layers = [
            name
            for name, _ in model.named_modules()
            if name not in model.task_dependent and name != ""
        ]
        _ = model.freeze_by_name(self._task_independent_layers)

    def on_train_epoch_start(self, trainer: Trainer, model: Model):
        if trainer.current_epoch == self.patience:
            _ = model.unfreeze_by_name(self._task_independent_layers)
