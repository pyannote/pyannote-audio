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

"""Resegmentation pipeline"""

from functools import partial
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Inference, Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import binarize


class Resegmentation(SpeakerDiarizationMixin, Pipeline):
    """Resegmentation pipeline

    This pipeline relies on a pretrained segmentation model to improve an existing diarization
    hypothesis. Resegmentation is done locally by sliding the segmentation model over the whole
    file. For each position of the sliding window, we find the optimal mapping between the input
    diarization and the output of the segmentation model and permutate the latter accordingly.
    Permutated local segmentations scores are then aggregated over time and postprocessed using
    hysteresis thresholding.

    It can also be used with `diarization` set to "annotation" to find a good estimate of optimal
    values for `onset`, `offset`, `min_duration_on`, and `min_duration_off` for any speaker
    diarization pipeline based on the `segmentation` model.

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".
    der_variant : dict, optional
        Optimize for a variant of diarization error rate.
        Defaults to {"collar": 0.0, "skip_overlap": False}. This is used in `get_metric`
        when instantiating the metric: GreedyDiarizationErrorRate(**der_variant).
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

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
        der_variant: dict = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.diarization = diarization

        model: Model = get_model(segmentation, use_auth_token=use_auth_token)
        self._segmentation = Inference(model)
        self._frames = self._segmentation.model.example_output.frames

        self._audio = model.audio

        # number of speakers in output of segmentation model
        self._num_speakers = len(model.specifications.classes)

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        # segmentation warm-up
        self.warm_up = Uniform(0.0, 0.1)

        # hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        # parameters optimized on DIHARD 3 development set
        if self.segmentation == "pyannote/segmentation":
            return {
                "warm_up": 0.05,
                "onset": 0.810,
                "offset": 0.481,
                "min_duration_on": 0.055,
                "min_duration_off": 0.098,
            }
        raise NotImplementedError()

    def classes(self):
        raise NotImplementedError()

    CACHED_SEGMENTATION = "cache/segmentation/inference"

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
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
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

        # binarize segmentations before speaker counting
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            initial_state=False,
        )

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            warm_up=(self.warm_up, self.warm_up),
            frames=self._frames,
        )
        hook("speaker_counting", count)

        # discretize original diarization
        # output shape is (num_frames, num_speakers)
        diarization = diarization or file[self.diarization]
        diarization = diarization.discretize(
            support=Segment(
                0.0, self._audio.get_duration(file) + self._segmentation.step
            ),
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

        # build discrete diarization
        discrete_diarization = self.to_diarization(permutated_segmentations, count)

        # convert to continuous diarization
        resegmentation = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        resegmentation.uri = file["uri"]

        # when reference is available, use it to map hypothesized speakers
        # to reference speakers (this makes later error analysis easier
        # but does not modify the actual output of the resegmentation pipeline)
        if "annotation" in file and file["annotation"]:
            resegmentation = self.optimal_mapping(file["annotation"], resegmentation)

        return resegmentation

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
