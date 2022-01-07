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
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Union, Optional, List, Dict, TYPE_CHECKING, Text

import numpy as np
from numba.typed import List
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform
from sortedcontainers import SortedDict

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from .utils import PipelineModel, get_devices, get_model
from ..utils.signal import Binarize

SpeakerClass = Text
MetaClasses = Dict[SpeakerClass, List[SpeakerClass]]

if TYPE_CHECKING:
    from ..tasks.segmentation.voice_type_classification import VoiceTypeClassification


@dataclass
class MultilabelDetectionSpecifications:
    classes: List[SpeakerClass]
    unions: MetaClasses
    intersections: MetaClasses
    unions_idx: Optional[SortedDict] = None
    intersections_idx: Optional[SortedDict] = None

    def __post_init__(self):
        # for each metaclass, mapping metaclass label to vector of its
        # classes's ids (used for encoding)
        self.unions_idx = self.to_metaclasses_idx(self.unions,
                                                  self.classes)
        self.intersections_idx = self.to_metaclasses_idx(self.intersections,
                                                         self.classes)

    @property
    def all_classes(self) -> List[str]:
        return (self.classes
                + list(self.unions.keys())
                + list(self.intersections.keys()))

    @staticmethod
    def to_metaclasses_idx(metaclasses: MetaClasses, classes: List[SpeakerClass]) -> SortedDict:
        return SortedDict({
            intersection_label: np.array([classes.index(klass)
                                          for klass in intersection_classes])
            for intersection_label, intersection_classes
            in metaclasses.items()
        })

    def derive_unions_encoding(self, one_hot_array: np.ndarray):
        arrays: List[np.ndarray] = []
        for label, idx in self.unions_idx.items():
            arrays.append(one_hot_array[:, idx].max(axis=1))
        return np.vstack(arrays).swapaxes(0, 1)

    def derive_intersections_encoding(self, one_hot_array: np.ndarray):
        arrays: List[np.ndarray] = []
        for label, idx in self.intersections_idx.items():
            arrays.append(one_hot_array[:, idx].min(axis=1))
        return np.vstack(arrays).swapaxes(0, 1)

    def derive_reference(self, annotation: Annotation) -> Annotation:
        derived = annotation.subset(self.classes)
        # Adding union labels
        for union_label, subclasses in self.unions.items():
            mapping = {k: union_label for k in subclasses}
            metalabel_annot = annotation.subset(union_label).rename_labels(mapping=mapping)
            derived.update(metalabel_annot.support())

        # adding intersection labels
        for intersect_label, subclasses in self.intersections.items():
            subclasses_tl = [annotation.label_timeline(subclass) for subclass in subclasses]
            overlap_tl = reduce(lambda x, y: x.crop(y), subclasses_tl)
            derived.update(overlap_tl.to_annotation(intersect_label))

        return derived

    @classmethod
    def from_parameters(
            cls,
            classes: List[SpeakerClass],  # VTC-specific parameter
            unions: Optional[MetaClasses] = None,
            intersections: Optional[MetaClasses] = None, ) \
            -> 'MultilabelDetectionSpecifications':
        if unions is not None:
            assert set(chain.from_iterable(unions.values())).issubset(set(classes))

        if intersections is not None:
            assert set(chain.from_iterable(intersections.values())).issubset(set(classes))

        classes = sorted(list(set(classes)))
        return cls(classes,
                   unions if unions else dict(),
                   intersections if intersections else dict())


class MultilabelFMeasure(BaseMetric):
    """Compute the mean Fscore over all labels

    """

    @classmethod
    def metric_name(cls):
        return "AVG[Labels]"

    def __init__(self, mtl_specs: MultilabelDetectionSpecifications,  # noqa
                 collar=0.0, skip_overlap=False,
                 beta=1., parallel=False, **kwargs):
        self.parallel = parallel
        self.metric_name_ = self.metric_name()
        self.components_ = set(self.metric_components())
        self.reset()
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.beta = beta
        self.mtl_specs = mtl_specs
        self.submetrics: Dict[str, DetectionPrecisionRecallFMeasure] = {
            label: DetectionPrecisionRecallFMeasure(collar=collar,
                                                    skip_overlap=skip_overlap,
                                                    beta=beta,
                                                    **kwargs)
            for label in self.mtl_specs.all_classes
        }

    def reset(self):
        super().reset()
        for submetric in self.submetrics.values():
            submetric.reset()

    def compute_components(self, reference: Annotation, hypothesis: Annotation, uem=None, **kwargs):

        details = self.init_components()
        reference = self.mtl_specs.derive_reference(reference)
        for label, submetric in self.submetrics.items():
            details[label] = submetric(reference=reference.subset([label]),
                                       hypothesis=hypothesis.subset([label]),
                                       uem=uem,
                                       **kwargs)
        return details

    def compute_metric(self, detail: Dict[str, float]):
        return np.mean(detail.values())

    def __abs__(self):
        return np.mean([abs(submetric) for submetric in self.submetrics.values()])


class MultilabelIER(IdentificationErrorRate):

    def __init__(self, mtl_specs: MultilabelDetectionSpecifications,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mtl_specs = mtl_specs

    def compute_components(self, reference, hypothesis, uem=None,
                           collar=None, skip_overlap=None, **kwargs):
        # deriving labels
        reference = self.mtl_specs.derive_reference(reference)
        return super().compute_components(reference, hypothesis,
                                          uem=uem, collar=collar,
                                          skip_overlap=skip_overlap,
                                          **kwargs)


class MultilabelDetection(Pipeline):
    """"""

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

        task: 'VoiceTypeClassification' = model.task
        self.mtl_specs = task.clsf_specs
        self.labels = task.clsf_specs.all_classes
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
        if self.training:
            if self.CACHED_ACTIVATIONS not in file:
                file[self.CACHED_ACTIVATIONS] = self.segmentation_inference_(file)
        else:
            file[self.CACHED_ACTIVATIONS] = self.segmentation_inference_(file)

        # for each class name, add
        multilabel_scores: SlidingWindowFeature = file[self.CACHED_ACTIVATIONS]
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
            label_annot = binarizer(label_scores)
            full_annot.update(label_annot)

        return full_annot

    def get_metric(self) -> Union[MultilabelFMeasure, IdentificationErrorRate]:
        """Return new instance of identification metric"""

        if self.fscore:
            return MultilabelFMeasure(mtl_specs=self.mtl_specs,
                                      collar=0.0, skip_overlap=False)
        else:
            return MultilabelIER(mtl_specs=self.mtl_specs,
                                 collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        else:
            return "minimize"
