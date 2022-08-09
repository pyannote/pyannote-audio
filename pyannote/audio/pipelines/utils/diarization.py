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

from typing import Mapping, Tuple, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import DiarizationErrorRate

from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.signal import Binarize, binarize


# TODO: move to dedicated module
class SpeakerDiarizationMixin:
    """Defines a bunch of methods common to speaker diarization pipelines"""

    @staticmethod
    def set_num_speakers(
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """Validate number of speakers

        Parameters
        ----------
        num_speakers : int, optional
            Number of speakers.
        min_speakers : int, optional
            Minimum number of speakers.
        max_speakers : int, optional
            Maximum number of speakers.

        Returns
        -------
        num_speakers : int or None
        min_speakers : int
        max_speakers : int or np.inf
        """

        # override {min|max}_num_speakers by num_speakers when available
        min_speakers = num_speakers or min_speakers or 1
        max_speakers = num_speakers or max_speakers or np.inf

        if min_speakers > max_speakers:
            raise ValueError(
                f"min_speakers must be smaller than (or equal to) max_speakers "
                f"(here: min_speakers={min_speakers:g} and max_speakers={max_speakers:g})."
            )
        if min_speakers == max_speakers:
            num_speakers = min_speakers

        return num_speakers, min_speakers, max_speakers

    @staticmethod
    def optimal_mapping(
        reference: Union[Mapping, Annotation], hypothesis: Annotation
    ) -> Annotation:
        """Find the optimal bijective mapping between reference and hypothesis labels

        Parameters
        ----------
        reference : Annotation or Mapping
            Reference annotation. Can be an Annotation instance or
            a mapping with an "annotation" key.
        hypothesis : Annotation

        Returns
        -------
        mapped : Annotation
            Hypothesis mapped to reference speakers.

        """
        if isinstance(reference, Mapping):
            reference = reference["annotation"]
            annotated = reference["annotated"] if "annotated" in reference else None
        else:
            annotated = None

        mapping = DiarizationErrorRate().optimal_mapping(
            reference, hypothesis, uem=annotated
        )
        return hypothesis.rename_labels(mapping=mapping)

    # TODO: get rid of onset/offset (binarization should be applied before calling speaker_count)
    # TODO: get rid of warm-up parameter (trimming should be applied before calling speaker_count)
    @staticmethod
    def speaker_count(
        segmentations: SlidingWindowFeature,
        onset: float = 0.5,
        offset: float = None,
        warm_up: Tuple[float, float] = (0.1, 0.1),
        frames: SlidingWindow = None,
    ) -> SlidingWindowFeature:
        """Estimate frame-level number of instantaneous speakers

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        onset : float, optional
           Onset threshold. Defaults to 0.5
        offset : float, optional
           Offset threshold. Defaults to `onset`.
        warm_up : (float, float) tuple, optional
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on
            `segmentations` shape and chunk size. Providing the exact frame
            resolution (when known) leads to better temporal precision.

        Returns
        -------
        count : SlidingWindowFeature
            (num_frames, 1)-shaped instantaneous speaker count
        """

        binarized: SlidingWindowFeature = binarize(
            segmentations, onset=onset, offset=offset, initial_state=False
        )
        trimmed = Inference.trim(binarized, warm_up=warm_up)
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames=frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        return count

    @staticmethod
    def to_annotation(
        discrete_diarization: SlidingWindowFeature,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> Annotation:
        """

        Parameters
        ----------
        discrete_diarization : SlidingWindowFeature
            (num_frames, num_speakers)-shaped discrete diarization
        min_duration_on : float, optional
            Defaults to 0.
        min_duration_off : float, optional
            Defaults to 0.

        Returns
        -------
        continuous_diarization : Annotation
            Continuous diarization
        """

        binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        return binarize(discrete_diarization)

    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build diarization out of preprocessed segmentation and precomputed speaker count

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped segmentations
        count : SlidingWindow_feature
            (num_frames, 1)-shaped speaker count

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        # TODO: investigate alternative aggregation
        activations = Inference.aggregate(
            segmentations,
            frames=count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )

        _, num_speakers = activations.data.shape
        count.data = np.minimum(count.data, num_speakers)

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0

        return SlidingWindowFeature(binary, activations.sliding_window)

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1
