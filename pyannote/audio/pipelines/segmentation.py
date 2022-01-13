# MIT License
#
# Copyright (c) 2018-2022 CNRS
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

"""Segmentation pipeline"""

from typing import Callable, Optional


from pyannote.core import SlidingWindowFeature
from pyannote.pipeline.parameter import Uniform

from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.core.inference import Inference
from pyannote.audio.core.model import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    get_devices,
    get_model,
)
from pyannote.audio.utils.signal import binarize
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate


class Segmentation(Pipeline):
    """Speaker segmentation

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    stitch : bool, optional
        Stitch adjacent chunks. Defaults to False

    Hyper-parameters
    ----------------
    onset : float
        Onset speaker activation threshold
    offset : float
        Offset speaker activation threshold
    
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        stitch: bool = False,
    ):
        super().__init__()

        if stitch:
            raise NotImplementedError("stitching is not yet supported")

        self.segmentation = segmentation
        self.stitch = stitch

        model: Model = get_model(segmentation)
        (device,) = get_devices(needs=1)
        model.to(device)
        self._segmentation = Inference(model)
        self._frames = self._segmentation.model.introspection.frames

        self._audio = model.audio

        # number of speakers in output of segmentation model
        self._num_speakers = len(model.specifications.classes)

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

    def default_parameters(self):
        # TODO: optimize those on DIHARD3
        return {"onset": 0.5, "offset": 0.5}

    CACHED_SEGMENTATION = "@segmentation/raw"

    def apply(
        self, file: AudioFile, hook: Optional[Callable] = None
    ) -> SlidingWindowFeature:

        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation(file)

        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        hook(self.CACHED_SEGMENTATION, segmentations)

        binarized: SlidingWindowFeature = binarize(
            segmentations, onset=self.onset, offset=self.offset, initial_state=False
        )

        return binarized

    def get_metric(self):
        return DiscreteDiarizationErrorRate()
