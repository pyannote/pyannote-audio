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

"""Segmentation pipelines"""

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class OracleSegmentation(Pipeline):
    """Oracle segmentation pipeline"""

    def apply(self, file: AudioFile) -> Annotation:
        """Return groundtruth segmentation

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Segmentation
        """

        return file["annotation"].relabel_tracks(generator="string")


class Segmentation(Pipeline):
    """Segmentation pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/Segmentation-PyanNet-DIHARD".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    batch_size : int, optional
        Batch size. Defaults to 32.

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speaker turn shorter than that many seconds.
    min_duration_off : float
        Fill same-speaker gaps shorter than that many seconds.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/Segmentation-PyanNet-DIHARD",
        batch_size: int = 32,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.batch_size = batch_size

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        self.segmentation_inference_ = Inference(
            model,
            batch_size=self.batch_size,
        )

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply segmentation

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        segmentation : `pyannote.core.Annotation`
            Segmentation
        """

        speaker_activations = self.segmentation_inference_(file)
        file["@segmentation/activations"] = speaker_activations
        segmentation = self._binarize(speaker_activations)
        segmentation.uri = file["uri"]
        return segmentation

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        return "minimize"
