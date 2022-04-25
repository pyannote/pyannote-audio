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

from typing import List, Mapping, Text, Union

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from pyannote.audio import Model


class GraduallyUnfreeze(Callback):
    """Gradually unfreeze layers

    1. Start training with all layers frozen, but those that depends on the task
       (i.e. those instantiated in model.build() and task.setup_loss_func()
    2. Train for a few epochs and unfreeze a few more layers
    3. Repeat

    Parameters
    ----------
    schedule:
        See examples for supported format.
    epochs_per_stage : int, optional
        Number of epochs between each stage. Defaults to 1.
        Has no effect if schedule is provided as a {layer_name: epoch} dictionary.

    Usage
    -----
    >>> callback = GraduallyUnfreeze()
    >>> Trainer(callbacks=[callback]).fit(model)

    Examples
    --------
    # for a model with PyanNet architecture (sincnet > lstm > linear > task_specific),
    # those are equivalent and will unfreeze 'linear' at epoch 1, 'lstm' at epoch 2,
    # and 'sincnet' at epoch 3.
    GraduallyUnfreeze()
    GraduallyUnfreeze(schedule=['linear', 'lstm', 'sincnet'])
    GraduallyUnfreeze(schedule={'linear': 1, 'lstm': 2, 'sincnet': 3})

    # the following syntax is also possible (with its dict-based equivalent just below):
    GraduallyUnfreeze(schedule=['linear', ['lstm', 'sincnet']], epochs_per_stage=10)
    GraduallyUnfreeze(schedule={'linear': 10, 'lstm': 20, 'sincnet': 20})
    # will unfreeze 'linear' at epoch 10, and both 'lstm' and 'sincnet' at epoch 20.
    """

    def __init__(
        self,
        schedule: Union[Mapping[Text, int], List[Union[List[Text], Text]]] = None,
        epochs_per_stage: int = None,
    ):
        super().__init__()

        if (
            (schedule is None) or (isinstance(schedule, List))
        ) and epochs_per_stage is None:
            epochs_per_stage = 1

        self.epochs_per_stage = epochs_per_stage
        self.schedule = schedule

    def on_fit_start(self, trainer: Trainer, model: Model):

        schedule = self.schedule

        task_specific_layers = model.task_dependent
        backbone_layers = [
            layer
            for layer, _ in reversed(ModelSummary(model, max_depth=1).named_modules)
            if layer not in task_specific_layers
        ]

        if schedule is None:
            schedule = backbone_layers

        if isinstance(schedule, List):
            _schedule = dict()
            for depth, layers in enumerate(schedule):
                layers = layers if isinstance(layers, List) else [layers]
                for layer in layers:
                    _schedule[layer] = (depth + 1) * self.epochs_per_stage
            schedule = _schedule

        self.schedule = schedule

        # freeze all but task specific layers
        for layer in backbone_layers:
            model.freeze_by_name(layer)

    def on_train_epoch_start(self, trainer: Trainer, model: Model):
        for layer, epoch in self.schedule.items():
            if epoch == trainer.current_epoch:
                model.unfreeze_by_name(layer)
