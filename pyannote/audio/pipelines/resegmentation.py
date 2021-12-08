# MIT License
#
# Copyright (c) 2018-2021 CNRS
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

"""Resegmentation pipeline"""

from typing import Callable, Optional, Text

import numpy as np

from pyannote.audio import Inference, Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class Resegmentation(SpeakerDiarizationMixin, Pipeline):
    """Resegmentation pipeline

    This pipeline relies on a pretrained segmentation model to improve an existing diarization
    hypothesis. Resegmentation is done locally by sliding the segmentation model over the whole
    file. For each position of the sliding window, we find the optimal mapping between the input
    diarization and the output of the segmentation model and permutate the latter accordingly.
    Permutated local segmentations scores are then aggregated over time and postprocessed using
    hysteresis thresholding.

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".

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
        segmentation: PipelineModel = "pyannote/segmentation",
        diarization: Text = "diarization",
    ):

        super().__init__()

        self.segmentation = segmentation
        self.diarization = diarization

        model: Model = get_model(segmentation)
        (device,) = get_devices(needs=1)
        model.to(device)
        self._segmentation = Inference(model)
        self._frames = self._segmentation.model.introspection.frames

        self._audio = model.audio

        # number of speakers in output of segmentation model
        self._num_speakers = len(model.specifications.classes)

        self.warm_up = 0.05

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    CACHED_SEGMENTATION = "@resegmentation/raw"

    def apply(
        self,
        file: AudioFile,
        diarization: Annotation = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        diarization : Annotation, optional
            Input diarization. Defaults to file[self.diarization].
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation(file)

        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        hook("@resegmentation/raw", segmentations)

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            warm_up=(self.warm_up, self.warm_up),
            frames=self._frames,
        )
        hook("@resegmentation/count", count)

        # discretize original diarization
        # output shape is (num_frames, num_speakers)
        diarization = diarization or file[self.diarization]
        diarization = diarization.discretize(
            support=Segment(0.0, self._audio.get_duration(file)),
            resolution=self._frames,
        )
        hook("@resegmentation/original", diarization)

        # remove warm-up regions from segmentation as they are less robust
        segmentations = Inference.trim(
            segmentations, warm_up=(self.warm_up, self.warm_up)
        )
        hook("@resegmentation/trim", segmentations)

        # zero-pad diarization or segmentation so they have the same number of speakers
        _, num_speakers = diarization.data.shape
        if num_speakers > self._num_speakers:
            segmentations.data = np.pad(
                segmentations.data,
                ((0, 0), (0, 0), (0, num_speakers - self._num_speakers)),
            )
        elif num_speakers < self._num_speakers:
            diarization.data = np.pad(
                diarization.data, ((0, 0), (0, self._num_speakers - num_speakers))
            )
            num_speakers = self._num_speakers

        # find optimal permutation with respect to the original diarization
        permutated_segmentations = np.full_like(segmentations.data, np.NAN)
        _, num_frames, _ = permutated_segmentations.shape
        for c, (chunk, segmentation) in enumerate(segmentations):
            local_diarization = diarization.crop(chunk)[np.newaxis, :num_frames]
            (permutated_segmentations[c],), _ = permutate(
                local_diarization,
                segmentation,
                cost_func=mae_cost_func,
            )
        permutated_segmentations = SlidingWindowFeature(
            permutated_segmentations, segmentations.sliding_window
        )
        hook("@resegmentation/permutated", permutated_segmentations)

        # reconstruct diarization
        diarization = self.to_diarization(
            permutated_segmentations,
            count,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        diarization.uri = file["uri"]

        if "annotation" in file:
            diarization = self.optimal_mapping(file["annotation"], diarization)

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
