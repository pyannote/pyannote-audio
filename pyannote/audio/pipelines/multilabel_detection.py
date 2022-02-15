# The MIT License (MIT)
#
# Copyright (c) 2017-2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Union, Dict, List

import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.tasks import MultilabelDetection
from .utils import PipelineModel, get_devices, get_model
from ..utils.signal import Binarize


class MultilabelFMeasure(BaseMetric):
    """
    Compute the mean Fscore over all labels
    """

    def metric_components(self):
        return self.classes

    @classmethod
    def metric_name(cls):
        return "AVG[Labels]"

    def __init__(self, classes: List[str],  # noqa
                 collar=0.0, skip_overlap=False,
                 beta=1., parallel=False, **kwargs):
        self.parallel = parallel
        self.metric_name_ = self.metric_name()
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.beta = beta
        self.classes = classes
        self.components_ = set(self.metric_components())

        self.submetrics: Dict[str, DetectionPrecisionRecallFMeasure] = {
            label: DetectionPrecisionRecallFMeasure(collar=collar,
                                                    skip_overlap=skip_overlap,
                                                    beta=beta,
                                                    **kwargs)
            for label in classes
        }

        self.reset()

    def reset(self):
        super().reset()
        for submetric in self.submetrics.values():
            submetric.reset()

    def compute_components(self, reference: Annotation, hypothesis: Annotation, uem=None, **kwargs):

        details = self.init_components()
        for label, submetric in self.submetrics.items():
            details[label] = submetric(reference=reference.subset([label]),
                                       hypothesis=hypothesis.subset([label]),
                                       uem=uem,
                                       **kwargs)
        return details

    def compute_metric(self, detail: Dict[str, float]):
        return np.mean(list(detail.values()))

    def report(self, display=False):
        df = super().report(display=False)

        for label, submetric in self.submetrics.items():
            df.loc["TOTAL"][label] = abs(submetric)

        if display:
            print(
                df.to_string(
                    index=True,
                    sparsify=False,
                    justify="right",
                    float_format=lambda f: "{0:.2f}".format(f),
                )
            )

        return df

    def __abs__(self):
        return np.mean([abs(submetric) for submetric in self.submetrics.values()])


class MultilabelDetectionPipeline(Pipeline):
    """Multilabel detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or multilabel detection) model.
        Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    fscore : bool, optional
        Optimize for average (precision/recall) fscore, over all classes.
        Defaults to optimizing identification error rate.
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference.

    Hyper-parameters
    ----------------

    For each class the pipeline is trained to detect, it works exactly
    like a VAD pipeline, and has the following hyper-parameters:

    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speech regions shorter than that many seconds.
    min_duration_off : float
        Fill non-speech regions shorter than that many seconds.
    """

    def __init__(self,
                 segmentation: PipelineModel = "pyannote/vtc",
                 fscore: bool = False,
                 **inference_kwargs,
                 ):

        super().__init__()

        self.segmentation = segmentation
        self.fscore = fscore

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        task: 'MultilabelDetection' = model.task
        self.labels = task.specifications.classes
        self.segmentation_inference_ = Inference(model, **inference_kwargs)

        self.binarize_hparams = ParamDict(**{
            class_name: ParamDict(
                onset=Uniform(0., 1.),
                offset=Uniform(0., 1.),
                min_duration_on=Uniform(0., 2.),
                min_duration_off=Uniform(0., 2.),
                pad_onset=Uniform(-1., 1.),
                pad_offset=Uniform(-1., 1.)
            ) for class_name in self.labels
        })

    def initialize(self):
        """Initialize pipeline with current set of parameters"""
        self.freeze({'binarize_hparams': {
            class_name: {
                "pad_onset": 0.0,
                "pad_offset": 0.0
            } for class_name in self.labels
        }})
        self._binarizers = {
            class_name: Binarize(
                onset=self.binarize_hparams[class_name]["onset"],
                offset=self.binarize_hparams[class_name]["offset"],
                min_duration_on=self.binarize_hparams[class_name]["min_duration_on"],
                min_duration_off=self.binarize_hparams[class_name]["min_duration_off"],
                pad_onset=self.binarize_hparams[class_name]["pad_onset"],
                pad_offset=self.binarize_hparams[class_name]["pad_offset"])
            for class_name in self.labels
        }

    CACHED_ACTIVATIONS = "@multilabel_detection/activations"

    def apply(self, file: AudioFile) -> Annotation:
        """Apply voice type classification

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Annotated classification.
        """

        multilabel_scores: SlidingWindowFeature
        if self.training:
            if self.CACHED_ACTIVATIONS not in file:
                file[self.CACHED_ACTIVATIONS] = self.segmentation_inference_(file)

            multilabel_scores = file[self.CACHED_ACTIVATIONS]
        else:
            multilabel_scores = self.segmentation_inference_(file)

        # for each class name, add class-specific "VAD" pipeline
        full_annot = Annotation(uri=file["uri"])
        for class_idx, class_name in enumerate(self.labels):
            # selecting scores for only one label
            label_scores_array: np.ndarray = multilabel_scores.data[:, class_idx]
            # creating a fake "num_classes" dim
            label_scores_array = np.expand_dims(label_scores_array, axis=1)
            # creating a new sliding window for that label
            label_scores = SlidingWindowFeature(label_scores_array,
                                                multilabel_scores.sliding_window)
            binarizer: Binarize = self._binarizers[class_name]
            class_annot = binarizer(label_scores)
            # cleaning up labels to the current detected label.
            class_annot.rename_labels({label: class_name for label in class_annot.labels()}, copy=False)
            full_annot.update(class_annot)

        return full_annot

    def get_metric(self) -> Union[MultilabelFMeasure, IdentificationErrorRate]:
        """Return new instance of identification metric"""

        if self.fscore:
            return MultilabelFMeasure(classes=self.labels, collar=0.0, skip_overlap=False)
        else:
            return IdentificationErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        else:
            return "minimize"
