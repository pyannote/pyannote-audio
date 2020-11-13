# MIT License
#
# Copyright (c) 2018-2020 CNRS
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

from typing import Text, Union

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform

from .speaker_change_detection import SpeakerChangeDetection
from .voice_activity_detection import VoiceActivityDetection


class SpeechTurnSegmentation(Pipeline):
    """Speech turn segmentation pipeline

    Combines voice activity and speaker change detections
    to obtain speaker-homogeneous speech regions.

    This pipeline is not optimizable because the metric to use for optimization
    remains an open problem. It can only be used as a part of larger pipeline.

    Parameters
    ----------
    vad_scores : Inference or str, optional
        `Inference` instance used to extract raw voice activity detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "vad".
    scd_scores : Inference or str, optional
        `Inference` instance used to extract raw speaker change detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "scd".

    Hyper-parameters
    ----------------
    gap_max_duration : float
        Since speaker change detection is trained to detect speaker/speaker changes
        but not speaker/non-speech changes, this pipeline considers that a speaker
        change should be added in the middle of every non-speech gap, unless the gap
        is shorter than `gap_max_duration` seconds AND speaker change did not
        actually detect a speaker change during the gap.

    Reference
    ---------
    https://gist.github.com/hbredin/caa5468b2b9f22a3ec6f650bfce060e5

    """

    def __init__(
        self,
        vad_scores: Union[Inference, Text] = "vad",
        scd_scores: Union[Inference, Text] = "scd",
    ):
        super().__init__()

        self.vad_scores = vad_scores
        self.scd_scores = scd_scores

        self.voice_activity_detection = VoiceActivityDetection(scores=self.vad_scores)
        self.speaker_change_detection = SpeakerChangeDetection(scores=self.scd_scores)

        self.gap_max_duration = Uniform(0.0, 2.0)

    def __call__(self, file: AudioFile) -> Annotation:
        """Apply speech turn segmentation

        Parameter
        ---------
        file : AudioFile
            Processed file.

        Returns
        -------
        speech_turns : Annotation
            Speech turns

        """

        vad = self.voice_activity_detection(file).get_timeline()
        scd = self.speaker_change_detection(file).get_timeline()

        # see https://gist.github.com/hbredin/caa5468b2b9f22a3ec6f650bfce060e5 for details
        return (
            scd.crop(vad.support(collar=self.gap_max_duration))
            .to_annotation(generator="string")
            .crop(vad)
        )
