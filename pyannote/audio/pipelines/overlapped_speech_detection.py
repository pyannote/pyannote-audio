# The MIT License (MIT)
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Overlapped speech detection pipelines"""

from functools import partial
from typing import Callable, Optional, Text, Union

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature, Timeline
from pyannote.database import get_annotated
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.signal import Binarize


def to_overlap(annotation: Annotation) -> Annotation:
    """Get overlapped speech regions

    Parameters
    ----------
    annotation : Annotation
        Speaker annotation.

    Returns
    -------
    overlap : Annotation
        Overlapped speech annotation.
    """

    overlap = Timeline(uri=annotation.uri)
    for (s1, t1), (s2, t2) in annotation.co_iter(annotation):
        l1 = annotation[s1, t1]
        l2 = annotation[s2, t2]
        if l1 == l2:
            continue
        overlap.add(s1 & s2)
    return overlap.support().to_annotation(generator="string", modality="overlap")


class OracleOverlappedSpeechDetection(Pipeline):
    """Oracle overlapped speech detection pipeline"""

    def apply(self, file: AudioFile) -> Annotation:
        """Return groundtruth overlapped speech detection

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : Annotation
            Overlapped speech regions.
        """
        return to_overlap(file["annotation"])


class OverlappedSpeechDetection(Pipeline):
    """Overlapped speech detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or overlapped speech detection) model.
        Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    precision : float, optional
        Optimize recall at target precision.
        Defaults to optimize precision/recall fscore.
    recall : float, optional
        Optimize precision at target recall
        Defaults to optimize precision/recall fscore
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`
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
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation

        # load model
        model = get_model(segmentation, use_auth_token=use_auth_token)

        if model.example_output.dimension > 1:
            inference_kwargs["pre_aggregation_hook"] = lambda scores: np.partition(
                scores, -2, axis=-1
            )[:, :, -2, np.newaxis]
        self._segmentation = Inference(model, **inference_kwargs)

        if model.specifications.powerset:
            self.onset = self.offset = 0.5

        else:
            #  hyper-parameters used for hysteresis thresholding
            self.onset = Uniform(0.0, 1.0)
            self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short overlapped regions
        # or filling short gaps between overlapped regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        if (precision is not None) and (recall is not None):
            raise ValueError(
                "One must choose between optimizing for target precision or target recall."
            )

        self.precision = precision
        self.recall = recall

    def default_parameters(self):
        if self.segmentation == "pyannote/segmentation":
            # parameters optimized on DIHARD 3 development set
            return {
                "onset": 0.430,
                "offset": 0.320,
                "min_duration_on": 0.091,
                "min_duration_off": 0.144,
            }

        elif self.segmentation == "pyannote/segmentation-3.0.0":
            return {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }

        raise NotImplementedError()

    def classes(self):
        return ["OVERLAP"]

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
        """Apply overlapped speech detection

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
        overlapped_speech : Annotation
            Overlapped speech regions.
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

        overlapped_speech = self._binarize(segmentations)
        overlapped_speech.uri = file["uri"]
        return overlapped_speech.rename_labels(
            {label: "OVERLAP" for label in overlapped_speech.labels()}
        )

    def get_metric(self, **kwargs) -> DetectionPrecisionRecallFMeasure:
        """Get overlapped speech detection metric

        Returns
        -------
        metric : DetectionPrecisionRecallFMeasure
            Detection metric.
        """

        if (self.precision is not None) or (self.recall is not None):
            raise NotImplementedError(
                "pyannote.pipeline should use `loss` method fallback."
            )

        class _Metric(DetectionPrecisionRecallFMeasure):
            def compute_components(
                _self,
                reference: Annotation,
                hypothesis: Annotation,
                uem: Timeline = None,
                **kwargs,
            ) -> dict:
                return super().compute_components(
                    to_overlap(reference), hypothesis, uem=uem, **kwargs
                )

        return _Metric()

    def loss(self, file: AudioFile, hypothesis: Annotation) -> float:
        """Compute recall at target precision (or vice versa)

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hypothesis : Annotation
            Hypothesized overlapped speech regions.

        Returns
        -------
        recall (or purity) : float
            When optimizing for target precision:
                If precision < target_precision, returns (precision - target_precision).
                If precision > target_precision, returns recall.
            When optimizing for target recall:
                If recall < target_recall, returns (recall - target_recall).
                If recall > target_recall, returns precision.
        """

        fmeasure = DetectionPrecisionRecallFMeasure()

        if "overlap_reference" in file:
            overlap_reference = file["overlap_reference"]

        else:
            reference = file["annotation"]
            overlap_reference = to_overlap(reference)
            file["overlap_reference"] = overlap_reference

        _ = fmeasure(overlap_reference, hypothesis, uem=get_annotated(file))
        precision, recall, _ = fmeasure.compute_metrics()

        if self.precision is not None:
            if precision < self.precision:
                return precision - self.precision
            else:
                return recall

        elif self.recall is not None:
            if recall < self.recall:
                return recall - self.recall
            else:
                return precision

    def get_direction(self):
        return "maximize"
