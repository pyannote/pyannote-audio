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

import functools
import os
from typing import Iterable

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.nn import Parameter
from torch.optim import Optimizer
from torch_audiomentations.utils.config import from_dict as get_augmentation

from pyannote.database import FileFinder, get_protocol


def get_optimizer(
    parameters: Iterable[Parameter], lr: float = 1e-3, cfg: DictConfig = None
) -> Optimizer:
    return instantiate(cfg.optimizer, parameters, lr=lr)


@hydra.main(config_path="train_config", config_name="config")
def main(cfg: DictConfig) -> None:

    # make sure to set the random seed before the instantiation of Trainer
    # so that each model initializes with the same weights when using DDP.
    seed = int(os.environ.get("PL_GLOBAL_SEED", "0"))
    seed_everything(seed=seed)

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    # TODO: configure scheduler
    # TODO: configure layer freezing

    # TODO: remove this OmegaConf.to_container hack once bug is solved:
    # https://github.com/omry/omegaconf/pull/443
    augmentation = (
        get_augmentation(OmegaConf.to_container(cfg.augmentation))
        if "augmentation" in cfg
        else None
    )

    optimizer = functools.partial(get_optimizer, cfg=cfg)

    task = instantiate(
        cfg.task,
        protocol,
        optimizer=optimizer,
        learning_rate=cfg.optimizer.lr,
        augmentation=augmentation,
    )

    callbacks = []

    model = instantiate(cfg.model, task=task)

    monitor, direction = task.val_monitor
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=None if monitor is None else 3,
        period=1,
        save_last=True,
        save_weights_only=False,
        dirpath=".",
        filename="{epoch}" if monitor is None else f"{{epoch}}-{{{monitor}:.6f}}",
        verbose=cfg.verbose,
    )
    callbacks.append(checkpoint)

    if monitor is not None:
        # TODO: add option to configure early stopping patience
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=direction,
            min_delta=0.0,
            patience=50,
            strict=True,
            verbose=cfg.verbose,
        )
        callbacks.append(early_stopping)

    # TODO: fail safely when log_graph=True raises an onnx error
    logger = TensorBoardLogger(
        ".",
        name="",
        version="",
        # log_graph=True,
    )

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model)

    # save paths to best models
    checkpoint.to_yaml()

    if monitor is not None:
        best_monitor = float(checkpoint.best_model_score)
        if direction == "min":
            return best_monitor
        else:
            return -best_monitor


if __name__ == "__main__":
    main()
