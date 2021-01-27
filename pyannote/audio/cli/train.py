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

from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="train_config", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    # make sure to set the random seed before the instantiation of Trainer
    # so that each model initializes with the same weights when using DDP.
    seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
    seed_everything(seed=seed)

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    # TODO: configure layer freezing

    # TODO: when fine-tuning or transfer learning a model whose last layers
    # needs to change -- fit those last layers for a bit before fitting the
    # whole model

    # TODO: remove this OmegaConf.to_container hack once bug is solved:
    # https://github.com/omry/omegaconf/pull/443
    augmentation = (
        get_augmentation(OmegaConf.to_container(cfg.augmentation))
        if "augmentation" in cfg
        else None
    )

    # instantiate task and model
    task = instantiate(cfg.task, protocol, augmentation=augmentation)
    model = instantiate(cfg.model, task=task)

    # setup optimizer and scheduler
    task.setup(stage="fit")
    model.setup(stage="fit")

    # TODO: catch resume_from_checkpoint before it is too late (i.e. before it overrides optimizer)

    model.optimizer = instantiate(cfg.optimizer, model.parameters())

    monitor, direction = task.val_monitor

    # TODO: allow configuring scheduler
    if monitor is None:
        model.scheduler = None
    else:
        model.scheduler = {
            "scheduler": ReduceLROnPlateau(
                model.optimizer,
                mode=direction,
                factor=0.1,
                patience=20,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=10,
                min_lr=0,
                eps=1e-08,
                verbose=False,
            ),
            "interval": "epoch",
            "reduce_on_plateau": True,
            "monitor": monitor,
            "strict": True,
        }

    callbacks = []

    if model.scheduler is not None:
        learning_rate_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(learning_rate_monitor)

    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=None if monitor is None else 3,
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
            patience=100,  # TODO: make it an hyper-parameter
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
