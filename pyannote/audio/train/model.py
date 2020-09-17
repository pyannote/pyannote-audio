#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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

"""Models

## Parts

>>> model.parts
["ff.1", "ff.2", "ff.3"]

## Probes

>>> model.probes = ["ff.1", "ff.2"]
>>> output, probes = model(input)
>>> ff1 = probes["ff.1"]
>>> ff2 = probes["ff.2"]

>>> del model.probes
>>> output = model(input)

## Freeze/unfreeze layers

>>> model.freeze(["ff.1", "ff.2"])
>>> model.unfreeze(["ff.2"])

"""

from typing import Union
from typing import List
from typing import Text
from typing import Tuple
from typing import Dict

try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal
from typing import Callable
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

RESOLUTION_FRAME = "frame"
RESOLUTION_CHUNK = "chunk"
Resolution = Union[SlidingWindow, Literal[RESOLUTION_FRAME, RESOLUTION_CHUNK]]

ALIGNMENT_CENTER = "center"
ALIGNMENT_STRICT = "strict"
ALIGNMENT_LOOSE = "loose"
Alignment = Literal[ALIGNMENT_CENTER, ALIGNMENT_STRICT, ALIGNMENT_LOOSE]

from pyannote.audio.train.task import Task
import numpy as np
import pescador
import torch
from torch.nn import Module
from functools import partial


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

    def __init__(self, specifications: dict, **architecture_params):
        super().__init__()
        self.specifications = specifications
        self.resolution_ = self.get_resolution(self.task, **architecture_params)
        self.alignment_ = self.get_alignment(self.task, **architecture_params)
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

    @property
    def probes(self):
        """Get list of probes"""
        return list(getattr(self, "_probes", []))

    @probes.setter
    def probes(self, names: List[Text]):
        """Set list of probes

        Parameters
        ----------
        names : list of string
            Names of modules to probe.
        """

        for handle in getattr(self, "handles_", []):
            handle.remove()

        self._probes = []

        if not names:
            return

        handles = []

        def _init(module, input):
            self.probed_ = dict()

        handles.append(self.register_forward_pre_hook(_init))

        def _append(name, module, input, output):
            self.probed_[name] = output

        for name, module in self.named_modules():
            if name in names:
                handles.append(module.register_forward_hook(partial(_append, name)))
                self._probes.append(name)

        def _return(module, input, output):
            return output, self.probed_

        handles.append(self.register_forward_hook(_return))

        self.handles_ = handles

    @probes.deleter
    def probes(self):
        """Remove all probes"""
        for handle in getattr(self, "handles_", []):
            handle.remove()
        self._probes = []

    @property
    def parts(self):
        """Names of (freezable / probable) modules"""
        return [n for n, _ in self.named_modules()]

    def freeze(self, names: List[Text]):
        """Freeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to freeze.
        """
        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = False

    def unfreeze(self, names: List[Text]):
        """Unfreeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to unfreeze.
        """

        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = True

    def forward(
        self, sequences: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[Text, torch.Tensor]]]:
        """TODO

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.Tensor`
        **kwargs : `dict`

        Returns
        -------
        output : (batch_size, ...) `torch.Tensor`
        probes : dict, optional
        """

        # TODO
        msg = "..."
        raise NotImplementedError(msg)

    @property
    def task(self) -> Task:
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications["task"]

    def get_resolution(self, task: Task, **architecture_params) -> Resolution:
        """Get frame resolution

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
        task : Task
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

        if task.returns_sequence:
            return RESOLUTION_FRAME

        elif task.returns_vector:
            return RESOLUTION_CHUNK

        else:
            # this should never happened
            msg = f"{task} tasks are not supported."
            raise NotImplementedError(msg)

    @property
    def resolution(self) -> Resolution:
        return self.resolution_

    def get_alignment(self, task: Task, **architecture_params) -> Alignment:
        """Get frame alignment

        This method is called by `BatchGenerator` instances to dermine how
        target tensors should be aligned with the output of the model.

        Default behavior is to return 'center'. In most cases, you should not
        need to worry about this but if you do, this method can be overriden to
        return 'strict' or 'loose'.

        Parameters
        ----------
        task : Task
        architecture_params : dict
            Architecture hyper-parameters.

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
        return self.specifications["X"]["dimension"]

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

        msg = f"{self.task} tasks do not define attribute 'dimension'."
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
            return self.specifications["y"]["classes"]

        msg = f"{self.task} tasks do not define attribute 'classes'."
        raise AttributeError(msg)

    def slide(
        self,
        features: SlidingWindowFeature,
        sliding_window: SlidingWindow,
        batch_size: int = 32,
        device: torch.device = None,
        skip_average: bool = None,
        postprocess: Callable[[np.ndarray], np.ndarray] = None,
        return_intermediate=None,
        progress_hook=None,
    ) -> SlidingWindowFeature:
        """Slide and apply model on features

        Parameters
        ----------
        features : SlidingWindowFeature
            Input features.
        sliding_window : SlidingWindow
            Sliding window used to apply the model.
        batch_size : int
            Batch size. Defaults to 32. Use large batch for faster inference.
        device : torch.device
            Device used for inference.
        skip_average : bool, optional
            For sequence labeling tasks (i.e. when model outputs a sequence of
            scores), each time step may be scored by several consecutive
            locations of the sliding window. Default behavior is to average
            those multiple scores. Set `skip_average` to False to return raw
            scores without averaging them.
        postprocess : callable, optional
            Function applied to the predictions of the model, for each batch
            separately. Expects a (batch_size, n_samples, n_features) np.ndarray
            as input, and returns a (batch_size, n_samples, any) np.ndarray.
        return_intermediate :
            Experimental. Not documented yet.
        progress_hook : callable
            Experimental. Not documented yet.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if skip_average is None:
            skip_average = (self.resolution == RESOLUTION_CHUNK) or (
                return_intermediate is not None
            )

        try:
            dimension = self.dimension
        except AttributeError:
            dimension = len(self.classes)

        resolution = self.resolution

        # model returns one vector per input frame
        if resolution == RESOLUTION_FRAME:
            resolution = features.sliding_window

        # model returns one vector per input window
        if resolution == RESOLUTION_CHUNK:
            resolution = sliding_window

        support = features.extent
        if support.duration < sliding_window.duration:
            chunks = [support]
            fixed = support.duration
        else:
            chunks = list(sliding_window(support, align_last=True))
            fixed = sliding_window.duration

        if progress_hook is not None:
            n_chunks = len(chunks)
            n_done = 0
            progress_hook(n_done, n_chunks)

        batches = pescador.maps.buffer_stream(
            iter(
                {"X": features.crop(window, mode="center", fixed=fixed)}
                for window in chunks
            ),
            batch_size,
            partial=True,
        )

        fX = []
        for batch in batches:

            tX = torch.tensor(batch["X"], dtype=torch.float32, device=device)

            # FIXME: fix support for return_intermediate
            with torch.no_grad():
                tfX = self(tX, return_intermediate=return_intermediate)

            tfX_npy = tfX.detach().to("cpu").numpy()
            if postprocess is not None:
                tfX_npy = postprocess(tfX_npy)

            fX.append(tfX_npy)

            if progress_hook is not None:
                n_done += len(batch["X"])
                progress_hook(n_done, n_chunks)

        fX = np.vstack(fX)

        if skip_average:
            return SlidingWindowFeature(fX, sliding_window)

        # get total number of frames (based on last window end time)
        n_frames = resolution.samples(chunks[-1].end, mode="center")

        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, dimension), dtype=np.float32)

        # k[i] is the number of chunks that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for chunk, fX_ in zip(chunks, fX):

            # indices of frames overlapped by chunk
            indices = resolution.crop(chunk, mode=self.alignment, fixed=fixed)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        return SlidingWindowFeature(data, resolution)
