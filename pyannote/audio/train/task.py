#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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

"""
Tasks
#####

This module provides a `Task` class meant to specify machine learning tasks
(e.g. classification or regression).

This may be used to infer parts of the network architecture and the associated
loss function automatically.

Example
-------
>>> voice_activity_detection = Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
...                                 output=TaskOutput.SEQUENCE)
"""

from enum import Enum
from typing import Union
from typing import NamedTuple
from typing import Callable


class TaskType(Enum):
    """Type of machine learning task

    Attributes
    ----------
    MULTI_CLASS_CLASSIFICATION
        multi-class classification
    MULTI_LABEL_CLASSIFICATION
        multi-label classification
    REGRESSION
        regression
    REPRESENTATION_LEARNING
        representation learning
    """

    MULTI_CLASS_CLASSIFICATION = 0
    MULTI_LABEL_CLASSIFICATION = 1
    REGRESSION = 2
    REPRESENTATION_LEARNING = 3


class TaskOutput(Enum):
    """Expected output

    Attributes
    ----------
    SEQUENCE
        A sequence of vector is expected.
    VECTOR
        A single vector is expected.
    """

    SEQUENCE = 0
    VECTOR = 1


class Task(NamedTuple):
    type: TaskType
    output: TaskOutput

    @classmethod
    def from_str(cls, representation: str):
        task_output, task_type = representation.split(" ", 1)

        if task_output == "frame-wise":
            task_output = TaskOutput.SEQUENCE

        elif task_output == "chunk-wise":
            task_output = TaskOutput.VECTOR

        else:
            msg = f'"{task_output}" task output is not supported.'
            raise NotImplementedError(msg)

        if task_type == "multi-class classification":
            task_type = TaskType.MULTI_CLASS_CLASSIFICATION

        elif task_type == "multi-label classification":
            task_type = TaskType.MULTI_LABEL_CLASSIFICATION

        elif task_type == "regression":
            task_type = TaskType.REGRESSION

        elif task_type == "representation learning":
            task_type = TaskType.REPRESENTATION_LEARNING

        else:
            msg = f'"{task_type}" task type is not supported.'
            raise NotImplementedError(msg)

        return cls(type=task_type, output=task_output)

    def __str__(self) -> str:
        """String representation"""

        if self.returns_sequence:
            name = "frame-wise"

        elif self.returns_vector:
            name = "chunk-wise"

        else:
            msg = (
                "string representation (__str__) is not implemented "
                "for this task output."
            )
            raise NotImplementedError(msg)

        if self.is_multiclass_classification:
            name = f"{name} multi-class classification"

        elif self.is_multilabel_classification:
            name = f"{name} multi-label classification"

        elif self.is_regression:
            name = f"{name} regression"

        elif self.is_representation_learning:
            name = f"{name} representation learning"

        else:
            msg = (
                "string representation (__str__) is not implemented "
                "for this type of task."
            )
            raise NotImplementedError(msg)

        return name

    @property
    def returns_sequence(self) -> bool:
        """Is the output expected to be a sequence?

        Returns
        -------
        `bool`
            `True` if the task output is a sequence, `False` otherwise.
        """
        return self.output == TaskOutput.SEQUENCE

    @property
    def returns_vector(self) -> bool:
        """Is the output expected to be a single vector?

        Returns
        -------
        `bool`
            `True` if the task output is a single vector, `False` otherwise.
        """
        return self.output == TaskOutput.VECTOR

    @property
    def is_multiclass_classification(self) -> bool:
        """Is it multi-class classification?

        Returns
        -------
        `bool`
            `True` if the task is multi-class classification
        """
        return self.type == TaskType.MULTI_CLASS_CLASSIFICATION

    @property
    def is_multilabel_classification(self) -> bool:
        """Is it multi-label classification?

        Returns
        -------
        `bool`
            `True` if the task is multi-label classification
        """
        return self.type == TaskType.MULTI_LABEL_CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """Is it regression?

        Returns
        -------
        `bool`
            `True` if the task is regression
        """
        return self.type == TaskType.REGRESSION

    @property
    def is_representation_learning(self) -> bool:
        """Is it representation learning?

        Returns
        -------
        `bool`
            `True` if the task is representation learning
        """
        return self.type == TaskType.REPRESENTATION_LEARNING

    @property
    def default_activation(self):
        """Default final activation

        Returns
        -------
        `torch.nn.LogSoftmax(dim=-1)` for multi-class classification
        `torch.nn.Sigmoid()` for multi-label classification
        `torch.nn.Identity()` for regression

        Raises
        ------
        NotImplementedError
            If the default activation cannot be guessed.
        """

        import torch.nn

        if self.is_multiclass_classification:
            return torch.nn.LogSoftmax(dim=-1)

        elif self.is_multilabel_classification:
            return torch.nn.Sigmoid()

        elif self.is_regression:
            return torch.nn.Identity()

        else:
            msg = f"Unknown default activation for {self} task."
            raise NotImplementedError(msg)
