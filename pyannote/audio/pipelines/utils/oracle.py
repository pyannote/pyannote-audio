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

from typing import Union

import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature

from pyannote.audio.core.io import Audio, AudioFile


def oracle_segmentation(
    file: AudioFile,
    window: SlidingWindow,
    frames: Union[SlidingWindow, float],
    num_speakers: int = None,
) -> SlidingWindowFeature:
    """Oracle speaker segmentation

    Simulates inference based on an (imaginary) oracle segmentation model:

    >>> oracle = Model.from_pretrained("oracle")
    >>> assert frames == oracle.example_output.frames
    >>> inference = Inference(oracle, duration=window.duration, step=window.step, skip_aggregation=True)
    >>> oracle_segmentation = inference(file)

    Parameters
    ----------
    file : AudioFile
        Audio file with "annotation".
    window : SlidingWindow
        Sliding window used for inference (see above)
    frames : SlidingWindow or float
        Output resolution of the oracle model (see above)
    num_speakers : int, optional
        Override the number of speakers returned by the oracle segmentation model
        Defaults to the actual number of speakers in the whole file

    Returns
    -------
    oracle_segmentation : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        Oracle segmentation.
    """

    if "duration" not in file:
        duration = Audio(mono="downmix").get_duration(file)
    else:
        duration: float = file["duration"]
    reference: Annotation = file["annotation"]

    if not isinstance(frames, SlidingWindow):
        frames = SlidingWindow(start=0.0, step=frames, duration=frames)

    labels = reference.labels()
    actual_num_speakers = len(labels)
    if num_speakers is None:
        num_speakers = actual_num_speakers

    if num_speakers > actual_num_speakers:
        num_missing = num_speakers - actual_num_speakers
        labels += [
            f"FakeSpeakerForOracleSegmentationInference{i:d}"
            for i in range(num_missing)
        ]

    window = SlidingWindow(start=0.0, duration=window.duration, step=window.step)

    segmentations = []
    for chunk in window(Segment(0.0, duration)):
        chunk_segmentation: SlidingWindowFeature = reference.discretize(
            chunk,
            resolution=frames,
            labels=labels,
            duration=window.duration,
        )

        if num_speakers < actual_num_speakers:
            # keep `num_speakers` most talkative speakers
            most_talkative_index = np.argsort(-np.sum(chunk_segmentation, axis=0))[
                :num_speakers
            ]
            chunk_segmentation = chunk_segmentation[:, most_talkative_index]

        segmentations.append(chunk_segmentation)

    return SlidingWindowFeature(np.float32(np.stack(segmentations)), window)
