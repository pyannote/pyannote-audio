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


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    #  TODO: configure augmentation
    #  TODO: configure scheduler

    # TODO: uncomment
    # optimizer = lambda parameters: instantiate(cfg.optimizer, parameters)
    # task = instantiate(cfg.task, protocol, optimizer=optimizer)
    task = instantiate(cfg.task, protocol)
    model = instantiate(cfg.model, task=task)

    save_dir = f"{protocol.name}"

    monitor, mode = task.validation_monitor
    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=10,
        period=1,
        save_last=True,
        save_weights_only=False,
        dirpath=save_dir,
        filename=f"{{epoch}}-{{{monitor}:.3f}}",
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=True,
    )

    # summary_writer_params = {
    #     "log_dir": None,
    #     "comment": "",
    #     "purge_step": None,
    #     "max_queue": 10,
    #     "flush_secs": 120,
    #     "filename_suffix": "",
    # }

    logger = TensorBoardLogger(
        save_dir,
        name="",
        version="",
        log_graph=True,
        # **summary_writer_params,
    )

    trainer = instantiate(
        cfg.trainer,
        callbacks=[model_checkpoint, early_stopping],
        logger=logger,
    )
    trainer.fit(model, task)


if __name__ == "__main__":
    main()
