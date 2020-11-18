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


from typing import Iterable

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Parameter
from torch.optim import Optimizer

from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="train_config", config_name="config")
def main(cfg: DictConfig) -> None:

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    # TODO: configure augmentation
    # TODO: configure scheduler
    # TODO: configure layer freezing

    def optimizer(parameters: Iterable[Parameter], lr: float = 1e-3) -> Optimizer:
        return instantiate(cfg.optimizer, parameters, lr=lr)

    task = instantiate(
        cfg.task,
        protocol,
        optimizer=optimizer,
        learning_rate=cfg.optimizer.lr,
    )

    model = instantiate(cfg.model, task=task)

    monitor, mode = task.validation_monitor
    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=10,
        period=1,
        save_last=True,
        save_weights_only=False,
        dirpath=".",
        filename=f"{{epoch}}-{{{monitor}:.3f}}",
        verbose=cfg.verbose,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=cfg.verbose,
    )

    logger = TensorBoardLogger(
        ".",
        name="",
        version="",
        log_graph=True,
    )

    trainer = instantiate(
        cfg.trainer,
        callbacks=[model_checkpoint, early_stopping],
        logger=logger,
    )

    if cfg.trainer.auto_lr_find == True:
        #  HACK: these two lines below should be removed once
        #  the corresponding bug is fixed in pytorch-lighting.
        #  https://github.com/pyannote/pyannote-audio/issues/514
        task.setup(stage="fit")
        model.setup(stage="fit")
        trainer.tune(model, task)

    trainer.fit(model, task)

    best_monitor = float(early_stopping.best_score)
    if mode == "min":
        return best_monitor
    else:
        return -best_monitor


if __name__ == "__main__":
    main()
