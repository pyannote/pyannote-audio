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

import math
from typing import Text

import numpy as np

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class BasicResegmentation(Pipeline):
    """Resegmentation pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/Segmentation-PyanNet-DIHARD".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".

    Hyper-parameters
    ----------------

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/Segmentation-PyanNet-DIHARD",
        batch_size: int = 32,
        diarization: Text = "diarization",
    ):

        super().__init__()

        self.segmentation = segmentation
        self.diarization = diarization
        self.batch_size = batch_size

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        self.audio_ = model.audio

        # duration of chunks (in seconds) given as input of segmentation model
        self.seg_chunk_duration_ = model.specifications.duration
        # number of speakers in output of segmentation model
        self.seg_num_speakers_ = len(model.specifications.classes)
        # duration of a frame (in seconds) in output of segmentation model
        self.seg_frame_duration_ = (
            model.introspection.inc_num_samples / self.audio_.sample_rate
        )
        # output frames as SlidingWindow instances
        self.seg_frames_ = SlidingWindow(
            start=0.0, step=self.seg_frame_duration_, duration=self.seg_frame_duration_
        )

        # prepare segmentation model for inference
        self.segmentation_inference_by_chunk_ = Inference(
            model,
            window="sliding",
            skip_aggregation=True,
            duration=self.seg_chunk_duration_,
            batch_size=32,
        )

        self.segmentation_inference_ = Inference(
            model,
            window="sliding",
            skip_aggregation=False,
            duration=self.seg_chunk_duration_,
            batch_size=32,
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
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # =====================================================================
        # Apply the pretrained segmentation model S on sliding chunks.
        # =====================================================================

        # output of segmentation model on each chunk
        segmentations: SlidingWindowFeature = self.segmentation_inference_by_chunk_(
            file
        )

        file["@debug/resegmentation/segmentation"] = self.segmentation_inference_(file)
        frames = file["@debug/resegmentation/segmentation"].sliding_window

        # number of frames in each chunk
        num_chunks, num_frames_in_chunk, num_speakers = segmentations.data.shape
        # number of frames in the whole file
        num_frames_in_file = math.ceil(
            self.audio_.get_duration(file) / self.seg_frame_duration_
        )

        # turn input diarization into binary (0 or 1) activations
        labels = file[self.diarization].labels()
        num_clusters = len(labels)
        y_original = np.zeros(
            (num_frames_in_file, len(labels)), dtype=segmentations.data.dtype
        )
        for k, label in enumerate(labels):
            segments = file[self.diarization].label_timeline(label)
            for start, stop in frames.crop(segments, mode="center", return_ranges=True):
                y_original[start:stop, k] += 1
        y_original = np.minimum(y_original, 1, out=y_original)
        diarization = SlidingWindowFeature(y_original, frames)
        file["@debug/resegmentation/diarization"] = diarization

        aggregated = np.zeros((num_frames_in_file, num_clusters))
        overlapped = np.zeros((num_frames_in_file, num_clusters))

        for chunk, segmentation in segmentations:

            # only consider active speakers in `segmentation`
            active = np.max(segmentation, axis=0) > self.onset
            if np.sum(active) == 0:
                continue
            segmentation = segmentation[:, active]

            # TODO/ understand why we have to do this :num_frames_in_chunk thing
            local_diarization = diarization.crop(chunk)[
                np.newaxis, :num_frames_in_chunk
            ]
            (permutated_segmentation,), (permutation,), (cost,) = permutate(
                local_diarization,
                segmentation,
                returns_cost=True,
            )

            start_frame = round(chunk.start / self.seg_frame_duration_)
            aggregated[
                start_frame : start_frame + num_frames_in_chunk
            ] += permutated_segmentation
            overlapped[start_frame : start_frame + num_frames_in_chunk] += 1.0

        speaker_activations = SlidingWindowFeature(
            aggregated / overlapped, frames, labels=labels
        )

        file["@debug/resegmentation/activations"] = speaker_activations

        return self._binarize(speaker_activations)

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
