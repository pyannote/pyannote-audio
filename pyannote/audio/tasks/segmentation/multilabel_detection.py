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
import logging
import warnings
from functools import reduce
from itertools import chain
from typing import List, Optional, Text, Tuple, Union, Dict

import numpy as np
from pyannote.core import Annotation
from pyannote.database import Protocol, ProtocolFile
from pyannote.database.protocol.protocol import Preprocessor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class VoiceTypeClassifierPreprocessor(Preprocessor):

    def __init__(self, classes: List[str],
                 unions: Optional[Dict[str, List[str]]] = None,
                 intersections: Optional[Dict[str, List[str]]] = None):
        self.classes = set(classes)
        if unions is not None:
            assert set(chain.from_iterable(unions.values())).issubset(set(classes))

        if intersections is not None:
            assert set(chain.from_iterable(intersections.values())).issubset(set(classes))
        self.unions = unions if unions is not None else dict()
        self.intersections = intersections if intersections is not None else dict()

    @property
    def all_classes(self) -> List[str]:
        return sorted(list(self.classes)
                      + list(self.unions.keys())
                      + list(self.intersections.keys()))

    def __call__(self, current_file: ProtocolFile) -> Annotation:
        annotation = current_file["annotation"]
        derived = annotation.subset(self.classes)
        # Adding union labels
        for union_label, subclasses in self.unions.items():
            mapping = {k: union_label for k in subclasses}
            metalabel_annot = annotation.subset(subclasses).rename_labels(mapping=mapping)
            derived.update(metalabel_annot.support())

        # adding intersection labels
        for intersect_label, subclasses in self.intersections.items():
            subclasses_tl = [annotation.label_timeline(subclass) for subclass in subclasses]
            overlap_tl = reduce(lambda x, y: x.crop(y), subclasses_tl)
            for seg in overlap_tl:
                derived[seg] = intersect_label

        return derived


class MultilabelDetection(SegmentationTaskMixin, Task):
    """Multilabel Detection

    Multilabel detection is the process of detecting when a specific class
    of speaker can be heard in the audio.
    It can also be used for speaker tracking.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"class_1", ...,
    "class_N"} where N is the number of classes in the training set.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    classes : List[str], optional
        list of classes that are to be detected. If unspecified, defaults to the
        all the classes contained in the training set.
    duration : float, optional
        Chunks duration. Defaults to 2s.
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: str, optional
        When provided, training samples are sampled uniformly with respect to that key.
        For instance, setting `balance` to "uri" will make sure that each file will be
        equally represented in the training samples.
    weight: str, optional
        When provided, use this key to as frame-wise weight in loss function.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
        Defaults to multiprocessing.cpu_count() // 2.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    augmentation : BaseWaveformTransform, optional
        torch_audiomentations waveform transform, used by dataloader
        during training.
    """

    ACRONYM = "mlt"

    def __init__(
            self,
            protocol: Protocol,
            classes: Optional[List[str]] = None,
            duration: float = 2.0,
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
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )

        self.balance = balance
        self.weight = weight
        self.classes = set(classes) if classes is not None else None

        # task specification depends
        # on the data: we do not know in advance which
        # classes should be detected. therefore, we postpone
        # the definition of specifications.

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)
        protocol_classes = set(self._train_metadata["annotation"])
        if self.classes is not None:
            if protocol_classes != self.classes:
                warnings.warn("Mismatch between protocol classes and classes "
                              f"passed to the task: {protocol_classes} != {self.classes}")
            classes = sorted(self.classes)
        else:
            classes = sorted(protocol_classes)

        logging.info(f"Classes for model: {classes}")
        self.specifications = Specifications(
            # one class per speaker
            classes=classes,
            # multiple speakers can be active at once
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
        )

    @property
    def chunk_labels(self) -> List[Text]:
        """Ordered list of labels

        Used by `prepare_chunk` so that y[:, k] corresponds to activity of kth speaker
        """
        return self.specifications.classes

    def prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Get speaker tracking targets"""
        return y

    # TODO: add option to give more weights to smaller classes
    # TODO: add option to balance training samples between classes
