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
from torch_audiomentations.utils.config import from_dict as get_augmentation

# from pyannote.audio.core.callback import GraduallyUnfreeze
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
    fine_tuning = cfg.model["_target_"] == "pyannote.audio.cli.pretrained"
    model = instantiate(cfg.model)
    model.task = task
    model.setup(stage="fit")

    # number of batches in one epoch
    num_batches_per_epoch = model.task.train__len__() // model.task.batch_size

    def configure_optimizers(self):

        optimizer = instantiate(cfg.optimizer, self.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer,
            monitor=monitor,
            direction=direction,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    model.configure_optimizers = MethodType(configure_optimizers, model)

    callbacks = []

    if fine_tuning:
        # TODO: for fine-tuning and/or transfer learning, we start by fitting
        # TODO: task-dependent layers and gradully unfreeze more layers
        # TODO: callbacks.append(GraduallyUnfreeze(epochs_per_stage=1))
        pass

    learning_rate_monitor = LearningRateMonitor()
    callbacks.append(learning_rate_monitor)

    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=None if monitor is None else 5,
        every_n_epochs=1,
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
            patience=cfg.lr_scheduler.patience * 2,
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

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # in case of fine-tuning, validate the initial model to make sure
    # that we actually improve over the initial performance
    if fine_tuning:
        trainer.validate(model)

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
