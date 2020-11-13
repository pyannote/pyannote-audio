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

"""Speaker change detection pipeline"""

from typing import Text, Union

from pyannote.audio.core.inference import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.utils.signal import Peak
from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform


class SpeakerChangeDetection(Pipeline):
    """Speaker change detection pipeline

    This pipeline is not optimizable because the metric to use for optimization
    remains an open problem. It can only be used as a part of larger pipeline.

    Parameters
    ----------
    scores : Inference or str, optional
        `Inference` instance used to extract raw speaker change detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "scd".

    Hyper-parameters
    ----------------
    alpha : float
        Peak detection threshold.
    min_duration : float
        Segment minimum duration.
    """

    def __init__(self, scores: Union[Inference, Text] = "scd"):
        super().__init__()

        self.scores = scores

        # hyper-parameters
        self.alpha = Uniform(0.0, 1.0)
        self.min_duration = Uniform(0.0, 10.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._peak = Peak(alpha=self.alpha, min_duration=self.min_duration)

    def __call__(self, file: AudioFile) -> Annotation:
        """Apply speaker change detection

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        speech_turns : pyannote.core.Annotation
            Speaker homogeneous regions.
        """

        if isinstance(self.scores, Inference):
            change_probability = self.scores(file)
        else:
            change_probability = file[self.scores]

        # peak detection
        change = self._peak(change_probability)
        change.uri = file.get("uri", None)
        return change.to_annotation(generator="string", modality="audio")
