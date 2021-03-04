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

import os
from types import MethodType
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_audiomentations.utils.config import from_dict as get_augmentation

from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="train_config", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    if cfg.trainer.get("resume_from_checkpoint", None) is not None:
        raise ValueError(
            "trainer.resume_from_checkpoint is not supported. "
            "use model=pretrained model.checkpoint=... instead."
        )

    # make sure to set the random seed before the instantiation of Trainer
    # so that each model initializes with the same weights when using DDP.
    seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
    seed_everything(seed=seed)

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    patience: int = cfg["patience"]

    # TODO: configure layer freezing

    # TODO: remove this OmegaConf.to_container hack once bug is solved:
    # https://github.com/omry/omegaconf/pull/443
    augmentation = (
        get_augmentation(OmegaConf.to_container(cfg.augmentation))
        if "augmentation" in cfg
        else None
    )

    # instantiate task and validation metric
    task = instantiate(cfg.task, protocol, augmentation=augmentation)
    monitor, direction = task.val_monitor

    # instantiate model
    pretrained = cfg.model["_target_"] == "pyannote.audio.cli.pretrained"
    model = instantiate(cfg.model, task=task)

    if not pretrained:
        # add task-dependent layers so that later call to model.parameters()
        # does return all layers (even task-dependent ones). this is already
        # done for pretrained models (TODO: check that this is true)
        task.setup(stage="fit")
        model.setup(stage="fit")

    def configure_optimizers(self):

        optimizer = instantiate(cfg.optimizer, self.parameters())

        if monitor is None:
            return optimizer

        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode=direction,
                factor=0.5,
                patience=4 * patience,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=2 * patience,
                min_lr=0,
                eps=1e-08,
                verbose=False,
            ),
            "interval": "epoch",
            "reduce_on_plateau": True,
            "monitor": monitor,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    model.configure_optimizers = MethodType(configure_optimizers, model)

    callbacks = []

    if pretrained:
        # for fine-tuning and/or transfer learning, we start by fitting
        # task-dependent layers and gradully unfreeze more layers
        callbacks.append(GraduallyUnfreeze(epochs_per_stage=patience))

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(learning_rate_monitor)

    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=None if monitor is None else 5,
        period=1,
        save_last=True,
        save_weights_only=False,
        dirpath=".",
        filename="{epoch}" if monitor is None else f"{{epoch}}-{{{monitor}:.6f}}",
        verbose=False,
    )
    callbacks.append(checkpoint)

    if monitor is not None:
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=direction,
            min_delta=0.0,
            patience=12 * patience,
            strict=True,
            verbose=False,
        )
        callbacks.append(early_stopping)

    logger = TensorBoardLogger(
        ".",
        name="",
        version="",
        log_graph=False,  # TODO: fixes onnx error with asteroid-filterbanks
    )

    # TODO: defaults to one-GPU training (one GPU is available)

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model)

    # save paths to best models
    checkpoint.to_yaml()

    # return the best validation score
    # this can be used for hyper-parameter optimization with Hydra sweepers
    if monitor is not None:
        best_monitor = float(checkpoint.best_model_score)
        if direction == "min":
            return best_monitor
        else:
            return -best_monitor


if __name__ == "__main__":
    main()
