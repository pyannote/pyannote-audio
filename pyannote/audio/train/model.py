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

from typing import Union, List, Text, Tuple, Dict, Optional, Callable

from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
from pyannote.core.utils.types import Alignment

from pyannote.audio.train.task import BaseTask
from pyannote.audio.train.task import Problem
from pyannote.audio.train.task import Resolution

import numpy as np
import pescador
import torch
import torch.nn as nn
from functools import partial


class Model(nn.Module):
    """Model

    A `Model` is nothing but a `torch.nn.Module` instance with a bunch of
    additional methods and properties specific to `pyannote.audio`.

    It is expected to be instantiated with a unique `task` positional argument
    describing the task addressed by the model, and a user-defined number of
    keyword arguments describing the model architecture.

    Parameters
    ----------
    task : BaseTask
        Task addressed by the model.
    **model_params : `dict`
        Model hyper-parameters.
    """

    def __init__(self, task: BaseTask, **model_hparams):
        super().__init__()

        # this hack is meant to avoid pytorch's submodule auto-registration loop
        # https://discuss.pytorch.org/t/avoid-the-side-effect-of-parameter-auto-registering-in-module/7238
        # BaseTask will register Model and Model will register Task

        self._task: List[BaseTask] = []
        self._task.append(task)

        self.init(**model_hparams)

    # see hack above
    @property
    def task(self) -> BaseTask:
        return self._task[0]

    def init(self, **model_hparams):
        """Initialize model architecture

        This method is called by Model.__init__ after attribute 'task' is set:
        this allows to access information about the task such as:
           - the input feature dimension (self.task.feature_extraction.dimension)
           - and many other details such self.task.problem, or
             self.task.resolution_{in|out}put

        Parameters
        ----------
        **model_hparams : `dict`
            Architecture hyper-parameters

        """
        msg = 'Method "init" must be overriden.'
        raise NotImplementedError(msg)

    def setup(self, stage: str):
        """Finalize model architecture

        This method is called after prepare_data.
        This allows to access information about the task such as:
           - training metadata (self.task.train_metadata)
           - the list of output classes (self.task.hparams.classes)
        """
        pass

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
        self, sequences: torch.Tensor,
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
    def resolution(self) -> Union[Resolution, SlidingWindow]:
        if not hasattr(self, "resolution_"):
            self.resolution_ = self.get_resolution()
        return self.resolution_

    def get_resolution(self) -> Union[Resolution, SlidingWindow]:
        """Get resolution of model output

        This method is called by the train dataloader to determine how target
        tensors should be built.

        Returns
        -------
        resolution: Resolution.CHUNK or SlidingWindow instance
            If resolution is Resolution.CHUNK, it means that the model returns
            just one output for the whole input chunk.
            If resolution is a SlidingWindow instances, it means that the model
            returns a sequence of frames.
        """
        return self.guess_resolution()

    def guess_resolution(self) -> Union[Resolution, SlidingWindow]:
        """Guess output resolution

        Returns
        -------
        resolution: Resolution.CHUNK or SlidingWindow instance
            Resolution.CHUNK if task.resolution_output is Resolution.CHUNK
            task.feature_extractoin_sliding_window otherwise.
        """

        if self.task.resolution_output == Resolution.CHUNK:
            return Resolution.CHUNK

        return self.task.feature_extraction.sliding_window

    @property
    def alignment(self) -> Alignment:
        if not hasattr(self, "alignment_"):
            self.alignment_ = self.get_alignment()
        return self.alignment_

    def get_alignment(self) -> Alignment:
        """Get model output frame alignment

        This method is called by the train dataloader to determine how target
        tenshors should be aligned with the model output.

        In most cases, you should not need to worry about this but if you do,
        this method can be overriden to return 'strict' or 'loose'.
        """
        return self.guess_alignment()

    def guess_alignment(self) -> Alignment:
        """Guess model output frame alignment"""
        return "center"

    def get_dimension(self) -> int:
        raise NotImplementedError()

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

        # if self.task["problem"] == Problem.REPRESENTATION:
        if self.task.problem == Problem.REPRESENTATION:
            return self.get_dimension()

        msg = "'dimension' is only defined for representation learning."
        raise AttributeError(msg)

    def slide(
        self,
        features: SlidingWindowFeature,
        sliding_window: SlidingWindow,
        batch_size: int = 32,
        device: Union[Text, torch.device] = None,
        skip_average: Optional[bool] = None,
        postprocess: Callable[[np.ndarray], np.ndarray] = None,
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
        progress_hook : callable
            Experimental. Not documented yet.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if skip_average is None:
            skip_average = self.resolution == Resolution.CHUNK

        try:
            dimension = self.dimension
        except AttributeError:
            dimension = len(self.task.hparams.classes)

        resolution = self.resolution

        # model returns one vector per input frame
        if resolution == Resolution.FRAME:
            resolution = features.sliding_window

        # model returns one vector per input window
        if resolution == Resolution.CHUNK:
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
                tfX = self(tX)

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
