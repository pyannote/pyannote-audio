# MIT License
#
# Copyright (c) 2020 CNRS
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

from typing import Callable, Iterable

import numpy as np
import scipy.signal
from torch.nn import Parameter
from torch.optim import Optimizer

from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database import Protocol


class SpeakerChangeDetection(SegmentationTaskMixin, Task):
    """Speaker change detection

    Speaker change detection is the task of detecting speaker change points
    in a given audio recording.

    Here, it is addressed with the same approach as voice activity detection,
    except "speech" class is replaced by "change" where a frame is marked as
    "change" if a speaker change happens less than `collar` frames away.

    Note that non-speech/speech changes are not marked as speaker change.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    collar : int, optional.
        Mark frames less than `collar` frames away from actual change point as positive.
        Defaults to 1.
    batch_size : int, optional
        Number of training samples per batch. Defaults to 32.
    num_workers : int, optional
        Number of workers used for generating training samples.
    pin_memory : bool, optional
        If True, data loaders will copy tensors into CUDA pinned
        memory before returning them. See pytorch documentation
        for more details. Defaults to False.
    optimizer : callable, optional
        Callable that takes model parameters as input and returns
        an Optimizer instance. Defaults to `torch.optim.Adam`.
    learning_rate : float, optional
        Learning rate. Defaults to 1e-3.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        collar: int = 1,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = False,
        optimizer: Callable[[Iterable[Parameter]], Optimizer] = None,
        learning_rate: float = 1e-3,
    ):

        super().__init__(
            protocol,
            duration=duration,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )

        self.specifications = TaskSpecification(
            problem=Problem.BINARY_CLASSIFICATION,
            scale=Scale.FRAME,
            duration=self.duration,
            classes=[
                "change",
            ],
        )

        self.collar = collar

    def prepare_y(self, one_hot_y: np.ndarray, collar: int = None):
        """Get speaker change detection targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.
        collar : int, optional
            Mark frames less than `collar` frames away from actual change point as positive.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is a change of speaker at tth frame, 0 otherwise.
        """

        if collar is None:
            collar = self.collar

        num_frames, num_speakers = one_hot_y.shape

        #  y[t] = True if speaker change, False otherwise
        y = np.sum(np.abs(np.diff(one_hot_y, axis=0)), axis=1, keepdims=True)
        y = np.vstack(([[0]], y > 0))

        # mark frames in the neighborhood of actual change point as positive.
        window = scipy.signal.triang(2 * collar + 1)[:, np.newaxis]
        y = np.minimum(1, scipy.signal.convolve(y, window, mode="same"))
        y = 1 * (y > 1e-10)

        # at this point, all segment boundaries are marked as change, including non-speech/speaker changes.
        # let's remove non-speech/speaker change

        # append empty samples at the beginning/end
        expanded_y = np.vstack(
            [
                np.zeros((collar, num_speakers), dtype=one_hot_y.dtype),
                one_hot_y,
                np.zeros((collar, num_speakers), dtype=one_hot_y.dtype),
            ]
        )

        # stride trick. data[i] is now a sliding window of collar length
        # centered at time step i.
        data = np.lib.stride_tricks.as_strided(
            expanded_y,
            shape=(num_frames, num_speakers, 2 * collar + 1),
            strides=(one_hot_y.strides[0], one_hot_y.strides[1], one_hot_y.strides[0]),
        )

        # y[i] = 1 if more than one speaker are speaking in the
        # corresponding window. 0 otherwise
        x_speakers = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
        x_speakers = x_speakers.reshape(-1, 1)

        y *= x_speakers

        return np.squeeze(y)
