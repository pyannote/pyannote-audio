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

"""Segmentation pipelines"""

from typing import Text, Union

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline
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
    scores : Inference or str, optional
        `Inference` instance used to extract raw segmentation scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "seg".

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speaker turn shorter than that many seconds.
    min_duration_off : float
        Fill same-speaker gaps shorter than that many seconds.
    last_active_patience : float
        Stop tracking a speaker if it has not been active for that many seconds.
        This hyper-parameter has no effect when optimizing the segmentation pipeline,
        but should be optimized when part of a larger (diarization) pipeline.
    """

    def __init__(self, scores: Union[Inference, Text] = "seg"):
        super().__init__()

        self.scores = scores

        # TODO / one binarize per speaker dimension

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 2.0)
        self.min_duration_off = Uniform(0.0, 2.0)

        # hyper-parameters that controls when to stop tracking a speaker.
        # this hyper-parameter has no effect when optimizing the Segmentation pipeline directly
        self.last_active_patience = Uniform(0.0, 2.0)

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

        if isinstance(self.scores, Inference):
            speakers_probability: SlidingWindowFeature = self.scores(file)
        else:
            speakers_probability = file[self.scores]

        sliding_window = speakers_probability.sliding_window

        uri = file.get("uri", None)
        segmentation = Annotation(uri=uri, modality="speech")

        previous_speaker_turn: Segment = None
        for i, data in enumerate(speakers_probability.data.T):
            speaker_probability = SlidingWindowFeature(
                data.reshape(-1, 1), sliding_window
            )
            for s, speaker_turn in enumerate(
                self._binarize(speaker_probability).get_timeline()
            ):
                if (s == 0) or (
                    speaker_turn.start - previous_speaker_turn.end
                    > self.last_active_patience
                ):
                    label = f"{i}-{s}"
                segmentation[speaker_turn, i] = label
                previous_speaker_turn = speaker_turn

        return segmentation.rename_labels(generator="string")

    def get_metric(self) -> GreedyDiarizationErrorRate:
        """Return new instance of segmentation metric"""

        # TODO: give each segment the same weight

        class _Metric(GreedyDiarizationErrorRate):
            def compute_components(
                _self,
                reference: Annotation,
                hypothesis: Annotation,
                uem: Timeline = None,
                **kwargs,
            ) -> dict:
                return super().compute_components(
                    reference.relabel_tracks(generator="string"),
                    hypothesis.relabel_tracks(generator="string"),
                    uem=uem,
                    **kwargs,
                )

        return _Metric()

    def get_direction(self):
        return "minimize"
