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
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})

    task = instantiate(cfg.task, protocol)
    model = instantiate(cfg.model, task=task)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, task)


if __name__ == "__main__":
    main()
