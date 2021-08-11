# MIT License
#
# Copyright (c) 2021 CNRS
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

import numpy as np
import torch

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model


class SpeakerEmbedding(Pipeline):
    """Speaker embedding pipeline

    This pipeline assumes that each file contains exactly one speaker
    and extracts one single embedding from the whole file.

    Parameters
    ----------
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/embedding".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        See pyannote.audio.pipelines.utils.get_model for supported format.
        Defaults to no voice activity detection.

    Usage
    -----
    >>> from pyannote.audio.pipelines import SpeakerEmbedding
    >>> pipeline = SpeakerEmbedding()
    >>> emb1 = pipeline("speaker1.wav")
    >>> emb2 = pipeline("speaker2.wav")
    >>> from scipy.spatial.distance import cdist
    >>> distance = cdist(emb1, emb2, metric="cosine")[0,0]
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        segmentation: PipelineModel = None,
    ):
        super().__init__()

        self.embedding = embedding
        self.segmentation = segmentation

        self.embedding_model_: Model = get_model(embedding)

        if self.segmentation is None:
            models = [self.embedding_model_]
        else:
            segmentation_model: Model = get_model(self.segmentation)
            models = [self.embedding_model_, segmentation_model]

        # send models to GPU (when GPUs are available and model is not already on GPU)
        cpu_models = [model for model in models if model.device.type == "cpu"]
        for cpu_model, gpu_device in zip(
            cpu_models, get_devices(needs=len(cpu_models))
        ):
            cpu_model.to(gpu_device)

        if self.segmentation is not None:
            self.voice_activity_ = Inference(
                segmentation_model,
                pre_aggregation_hook=lambda scores: np.max(
                    scores, axis=-1, keepdims=True
                ),
            )

    def apply(self, file: AudioFile) -> np.ndarray:

        device = self.embedding_model_.device

        # read audio file and send it to GPU
        waveform = self.embedding_model_.audio(file)[0][None].to(device)

        if self.segmentation is None:
            weights = None
        else:
            # obtain voice activity scores
            weights = self.voice_activity_(file).data
            # HACK -- this should be fixed upstream
            weights[np.isnan(weights)] = 0.0
            weights = torch.from_numpy(weights ** 3)[None, :, 0].to(device)

        # extract speaker embedding on parts of
        with torch.no_grad():
            return self.embedding_model_(waveform, weights=weights).cpu().numpy()


def main(
    protocol: str = "VoxCeleb.SpeakerVerification.VoxCeleb1",
    subset: str = "test",
    embedding: str = "pyannote/embedding",
    segmentation: str = None,
):

    import typer
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

    from pyannote.database import FileFinder, get_protocol
    from pyannote.metrics.binary_classification import det_curve

    pipeline = SpeakerEmbedding(embedding=embedding, segmentation=segmentation)

    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})

    y_true, y_pred = [], []

    emb = dict()

    trials = getattr(protocol, f"{subset}_trial")()

    for t, trial in enumerate(tqdm(trials)):

        audio1 = trial["file1"]["audio"]
        if audio1 not in emb:
            emb[audio1] = pipeline(audio1)

        audio2 = trial["file2"]["audio"]
        if audio2 not in emb:
            emb[audio2] = pipeline(audio2)

        y_pred.append(cdist(emb[audio1], emb[audio2], metric="cosine")[0][0])
        y_true.append(trial["reference"])

    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    typer.echo(
        f"{protocol.name} | {subset} | {embedding} | {segmentation} | EER = {100 * eer:.3f}%"
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
