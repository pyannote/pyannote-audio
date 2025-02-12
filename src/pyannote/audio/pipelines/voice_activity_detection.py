# MIT License
#
# Copyright (c) 2018- CNRS
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

"""Voice activity detection pipelines"""

from functools import partial
from pathlib import Path
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.signal import Binarize


class OracleVoiceActivityDetection(Pipeline):
    """Oracle voice activity detection pipeline"""

    @staticmethod
    def apply(file: AudioFile) -> Annotation:
        """Return groundtruth voice activity detection

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speech regions
        """

        speech = file["annotation"].get_timeline().support()
        return speech.to_annotation(generator="string", modality="speech")


class VoiceActivityDetection(Pipeline):
    """Voice activity detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.
    token : str or bool, optional
        Huggingface token to be used for downloading from Huggingface hub.
    cache_dir: Path or str, optional
        Path to the folder where files downloaded from Huggingface hub are stored.
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference.

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speech regions shorter than that many seconds.
    min_duration_off : float
        Fill non-speech regions shorter than that many seconds.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        token: Union[Text, None] = None,
        cache_dir: Union[Path, Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.fscore = fscore

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation, token=token, cache_dir=cache_dir)

        inference_kwargs["pre_aggregation_hook"] = lambda scores: np.max(
            scores, axis=-1, keepdims=True
        )
        self._segmentation = Inference(model, **inference_kwargs)

        if model.specifications.powerset:
            self.onset = self.offset = 0.5
        else:
            #  hyper-parameters used for hysteresis thresholding
            self.onset = Uniform(0.0, 1.0)
            self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech regions
        # or filling short gaps between speech regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        if self.segmentation == "pyannote/segmentation":
            # parameters optimized for DIHARD 3 development set
            return {
                "onset": 0.767,
                "offset": 0.377,
                "min_duration_on": 0.136,
                "min_duration_off": 0.067,
            }

        elif self.segmentation == "pyannote/segmentation-3.0.0":
            return {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }

        raise NotImplementedError()

    def classes(self):
        return ["SPEECH"]

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, hook=partial(hook, "segmentation", None)
                )
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(
                file, hook=partial(hook, "segmentation", None)
            )

        hook("segmentation", segmentations)

        speech: Annotation = self._binarize(segmentations)
        speech.uri = file["uri"]
        return speech.rename_labels({label: "SPEECH" for label in speech.labels()})

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"
