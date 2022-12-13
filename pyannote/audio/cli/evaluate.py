# MIT License
#
# Copyright (c) 2022- CNRS
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


from typing import Optional

import hydra
from omegaconf import DictConfig
from pyannote.database import FileFinder, ProtocolFile, get_protocol
from rich.progress import Progress

from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
from pyannote.audio.utils.signal import binarize


@hydra.main(config_path="evaluate_config", config_name="config")
def evaluate(cfg: DictConfig) -> Optional[float]:

    # load pretrained model
    (device,) = get_devices(needs=1)
    model = Model.from_pretrained(cfg.model, device=device)

    # load evaluation files
    protocol = get_protocol(cfg.protocol, preprocessors={"audio": FileFinder()})
    files = list(getattr(protocol, cfg.subset)())

    # load evaluation metric
    metric = DiscreteDiarizationErrorRate()

    with Progress() as progress:

        main_task = progress.add_task(protocol.name, total=len(files))
        file_task = progress.add_task("Processing", total=1.0)

        def progress_hook(completed: int = None, total: int = None):
            progress.update(file_task, completed=completed / total)

        inference = Inference(model, device=device)
        warm_up = cfg.warm_up / inference.duration

        def hypothesis(file: ProtocolFile):
            return Inference.trim(
                binarize(inference(file, hook=progress_hook)),
                warm_up=(warm_up, warm_up),
            )

        metric = DiscreteDiarizationErrorRate()

        for file in files:
            progress.update(file_task, description=file["uri"])
            reference = file["annotation"]
            uem = file["annotated"]
            _ = metric(reference, hypothesis(file), uem=uem)
            progress.advance(main_task)

    report = metric.report(display=False)

    with open("report.txt", "w") as f:

        f.write(f"# Model:    {cfg.model}\n")
        f.write(f"# Protocol: {protocol.name}\n")
        f.write(f"# Subset:   {cfg.subset}\n")
        f.write("\n")
        report = report.to_string(
            index=True,
            sparsify=False,
            justify="right",
            float_format=lambda f: "{0:.2f}".format(f),
        )
        f.write(f"{report}")


if __name__ == "__main__":
    evaluate()
