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
TODO
"""

from typing import Union
from typing import List
try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal
from pyannote.core import SlidingWindow

RESOLUTION_FRAME = 'frame'
RESOLUTION_CHUNK = 'chunk'
Resolution = Union[SlidingWindow, Literal[RESOLUTION_FRAME, RESOLUTION_CHUNK]]

ALIGNMENT_CENTER = 'center'
ALIGNMENT_STRICT = 'strict'
ALIGNMENT_LOOSE = 'loose'
Alignment = Literal[ALIGNMENT_CENTER, ALIGNMENT_STRICT, ALIGNMENT_LOOSE]

from pyannote.audio.train.task import Task
from torch.nn import Module


class Model(Module):
    """Model

    A `Model` is nothing but a `torch.nn.Module` instance with a bunch of
    additional methods and properties specific to `pyannote.audio`.

    It is expected to be instantiated with a unique `specifications` positional
    argument describing the task addressed by the model, and a user-defined
    number of keyword arguments describing the model architecture.

    Parameters
    ----------
    specifications : `dict`
        Task specifications.
    **architecture_params : `dict`
        Architecture hyper-parameters.
    """

    def __init__(self,
                 specifications: dict,
                 **architecture_params):
        super().__init__()
        self.specifications = specifications
        self.resolution_ = self.get_resolution(**architecture_params)
        self.alignment_ = self.get_alignment(**architecture_params)
        self.init(**architecture_params)

    def init(self, **architecture_params):
        """Initialize model architecture

        This method is called by Model.__init__ after attributes
        'specifications', 'resolution_', and 'alignment_' have been set.

        Parameters
        ----------
        **architecture_params : `dict`
            Architecture hyper-parameters

        """
        msg = 'Method "init" must be overriden.'
        raise NotImplementedError(msg)

    def forward(self, sequences, **kwargs):
        """TODO

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.Tensor`
        **kwargs : `dict`

        Returns
        -------
        output : (batch_size, ...) `torch.Tensor`
        """

        # TODO
        msg = "..."
        raise NotImplementedError(msg)

    @property
    def task(self) -> Task:
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications['task']

    def get_resolution(self, **architecture_params) -> Resolution:
        """Get target resolution

        This method is called by `BatchGenerator` instances to determine how
        target tensors should be built.

        Depending on the task and the architecture, the output of a model will
        have different resolution. The default behavior is to return
        - `RESOLUTION_CHUNK` if the model returns just one output for the whole
          input sequence
        - `RESOLUTION_FRAME` if the model returns one output for each frame of
          the input sequence

        In case neither of these options is valid, this method needs to be
        overriden to return a custom `SlidingWindow` instance.

        Parameters
        ----------
        **architecture_params
            Parameters used for instantiating the model architecture.

        Returns
        -------
        resolution : `Resolution`
            - `RESOLUTION_CHUNK` if the model returns one single output for the
              whole input sequence;
            - `RESOLUTION_FRAME` if the model returns one output for each frame
               of the input sequence.
        """

        if self.task.returns_sequence:
            return RESOLUTION_FRAME

        elif self.task.returns_vector:
            return RESOLUTION_CHUNK

        else:
            # this should never happened
            msg = f"{self.task} tasks are not supported."
            raise NotImplementedError(msg)

    @property
    def resolution(self) -> Resolution:
        return self.resolution_

    def get_alignment(self, **architecture_params) -> Alignment:
        """Get target alignment

        This method is called by `BatchGenerator` instances to dermine how
        target tensors should be aligned with the output of the model.

        Default behavior is to return 'center'. In most cases, you should not
        need to worry about this but if you do, this method can be overriden to
        return 'strict' or 'loose'.

        Returns
        -------
        alignment : `Alignment`
            Target alignment. Must be one of 'center', 'strict', or 'loose'.
            Always returns 'center'.
        """

        return ALIGNMENT_CENTER

    @property
    def alignment(self) -> Alignment:
        return self.alignment_

    @property
    def n_features(self) -> int:
        """Number of input features

        Shortcut for self.specifications['X']['dimension']

        Returns
        -------
        n_features : `int`
            Number of input features
        """
        return self.specifications['X']['dimension']

    @property
    def dimension(self) -> int:
        """Output dimension

        This method needs to be overriden for representation learning tasks,
        because output dimension cannot be inferred from the task
        specifications.

        Returns
        -------
        dimension : `int`
            Dimension of model output.

        Raises
        ------
        AttributeError
            If the model addresses a classification or regression task.
        """

        if self.task.is_representation_learning:
            msg = (
                f"Class {self.__class__.__name__} needs to define "
                f"'dimension' property."
            )
            raise NotImplementedError(msg)

        msg = (f"{self.task} tasks do not define attribute 'dimension'.")
        raise AttributeError(msg)

    @property
    def classes(self) -> List[str]:
        """Names of classes

        Shortcut for self.specifications['y']['classes']

        Returns
        -------
        classes : `list` of `str`
            List of names of classes.


        Raises
        ------
        AttributeError
            If the model does not address a classification task.
        """

        if not self.task.is_representation_learning:
            return self.specifications['y']['classes']

        msg = (f"{self.task} tasks do not define attribute 'classes'.")
        raise AttributeError(msg)
