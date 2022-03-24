#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2022- CNRS

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

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from functools import reduce
from itertools import chain
from typing import Dict, List, Optional, Set

from pyannote.core import Annotation, Segment
from pyannote.database import ProtocolFile


class LowerTemporalResolution:
    """Artificially degrade temporal resolution of reference annotation

    Parameters
    ----------
    resolution : float, optional
        Target temporal resolution, in seconds. Defaults to 0.1 (100ms).
    """

    preprocessed_key = "annotation"

    def __init__(self, resolution: float = 0.1):
        super().__init__()
        self.resolution = resolution

    def __call__(self, current_file: ProtocolFile) -> Annotation:

        annotation = current_file["annotation"]
        new_annotation = annotation.empty()

        for new_track, (segment, _, label) in enumerate(
            annotation.itertracks(yield_label=True)
        ):
            new_start = self.resolution * int(segment.start / self.resolution + 0.5)
            new_end = self.resolution * int(segment.end / self.resolution + 0.5)
            new_segment = Segment(start=new_start, end=new_end)
            new_annotation[new_segment, new_track] = label

        support = current_file["annotated"].extent()
        return new_annotation.support().crop(support)


class DeriveMetaLabels:
    """Filters out classes not defined in the `classes` list and
    synthesizes additional classes based on unions or intersection of classes.

    Parameters
    ----------
    classes : List[str]
        All the "base" classes that should be used in the protocol's annotation's
    unions: Dict[str, List[str]], optional
        Unions of classes. The keys are the name of the new union classes, and the values are the
        list of classes that should used for these unions.
    intersections: Dict[str, List[str]], optional
        Intersections of classes. The keys are the name of the new intersections classes, and the values are the
        list of classes that should used for these intersections.
    """

    def __init__(
        self,
        classes: List[str],
        unions: Optional[Dict[str, List[str]]] = None,
        intersections: Optional[Dict[str, List[str]]] = None,
    ):
        self.classes: Set[str] = set(classes)
        self.unions = unions if unions is not None else dict()
        self.intersections = intersections if intersections is not None else dict()

    @property
    def all_classes(self) -> List[str]:
        """A list of all the classes (base, union-based and intersection-based) that can be found
        in output annotations from this preprocessor"""
        return sorted(
            list(self.classes)
            + list(self.unions.keys())
            + list(self.intersections.keys())
        )

    def __call__(self, current_file: ProtocolFile) -> Annotation:
        annotation: Annotation = current_file["annotation"]
        derived = annotation.subset(self.classes)
        # Adding union labels
        for union_label, subclasses in self.unions.items():
            # creates a subset of the original annotation, based
            mapping = {k: union_label for k in subclasses}
            metalabel_annot = annotation.subset(subclasses).rename_labels(
                mapping=mapping
            )
            derived.update(metalabel_annot.support())

        # adding intersection labels
        for intersect_label, subclasses in self.intersections.items():
            # a bit trickier: for each intersection meta-class's subclass,
            # we retrieve its timeline
            subclasses_tl = [
                annotation.label_timeline(subclass) for subclass in subclasses
            ]
            # then we iteratively re-crop each of the timelines one onto
            # the other
            overlap_tl = reduce(lambda x, y: x.crop(y), subclasses_tl)
            for seg in overlap_tl:
                derived[seg] = intersect_label

        return derived
