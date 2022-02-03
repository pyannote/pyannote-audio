# MIT License
#
# Copyright (c) 2020-2021 CNRS
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
from typing import Tuple, Union, Optional, Text, List

import numpy as np
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from .mixins import SegmentationTaskMixin
from ...core.task import Task, Specifications, Problem, Resolution
from ...pipelines.multilabel_detection import MultilabelDetectionSpecifications, SpeakerClass, MetaClasses


class VoiceTypeClassification(SegmentationTaskMixin, Task):
    """"""

    ACRONYM = "vtc"

    def __init__(
            self,
            protocol: Protocol,
            classes: List[SpeakerClass],  # VTC-specific parameter
            unions: Optional[MetaClasses] = None,
            intersections: Optional[MetaClasses] = None,
            duration: float = 5.0,
            warm_up: Union[float, Tuple[float, float]] = 0.0,
            balance: Text = None,
            weight: Text = None,
            batch_size: int = 32,
            num_workers: int = None,
            pin_memory: bool = False,
            augmentation: BaseWaveformTransform = None,
    ):
        super().__init__(
            protocol,
            duration=duration,
            min_duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )
        self.balance = balance
        self.weight = weight

        self.clsf_specs = MultilabelDetectionSpecifications. \
            from_parameters(classes, unions, intersections)

        # setting up specifications, used to set up the model by pt-lightning
        self.specifications = Specifications(
            # it is a multi-label classification problem
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            # we expect the model to output one prediction
            # for the whole chunk
            resolution=Resolution.FRAME,
            # the model will ingest chunks with that duration (in seconds)
            duration=self.duration,
            # human-readable names of classes
            classes=self.clsf_specs.all_classes
        )

    @property
    def chunk_labels(self) -> List[SpeakerClass]:
        # Only used by `prepare_chunk`, thus, which doesn't need to know
        # about union/intersections.
        return self.clsf_specs.classes

    def prepare_y(self, one_hot_y: np.ndarray) -> np.ndarray:
        # one_hot_y is of shape (Time, Classes)
        metaclasses_one_hots = []
        if self.clsf_specs.unions:
            metaclasses_one_hots.append(self.clsf_specs.derive_unions_encoding(one_hot_y))
        if self.clsf_specs.intersections:
            metaclasses_one_hots.append(self.clsf_specs.derive_intersections_encoding(one_hot_y))

        if metaclasses_one_hots:
            one_hot_y = np.hstack([one_hot_y] + metaclasses_one_hots)
        return np.int64(one_hot_y)
