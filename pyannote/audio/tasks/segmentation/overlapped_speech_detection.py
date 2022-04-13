# MIT License
#
# Copyright (c) 2020- CNRS
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


from typing import Dict, Sequence, Text, Tuple, Union

import torch
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin


class OverlappedSpeechDetection(SegmentationTaskMixin, Task):
    """Overlapped speech detection

    Overlapped speech detection is the task of detecting regions where at least
    two speakers are speaking at the same time.

    Here, it is addressed with the same approach as voice activity detection,
    except "speech" class is replaced by "overlap", where a frame is marked as
    "overlap" if two speakers or more are active.

    Note that data augmentation is used to increase the proporition of "overlap".
    This is achieved by generating chunks made out of the (weighted) sum of two
    random chunks.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
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
    overlap: dict, optional
        Controls how artificial chunks with overlapping speech are generated:
        - "probability" key is the probability of artificial overlapping chunks. Setting
          "probability" to 0.6 means that, on average, 40% of training chunks are "real"
          chunks, while 60% are artifical chunks made out of the (weighted) sum of two
          chunks. Defaults to 0.5.
        - "snr_min" and "snr_max" keys control the minimum and maximum signal-to-noise
          ratio between summed chunks, in dB. Default to 0.0 and 10.
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
    metric : optional
        Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
        Defaults to AUROC (area under the ROC curve).
    """

    OVERLAP_DEFAULTS = {"probability": 0.5, "snr_min": 0.0, "snr_max": 10.0}

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        overlap: dict = OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):

        super().__init__(
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

        self.specifications = Specifications(
            problem=Problem.BINARY_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            warm_up=self.warm_up,
            classes=[
                "overlap",
            ],
        )

        self.overlap = overlap
        self.balance = balance
        self.weight = weight

    def adapt_y(self, collated_y: torch.Tensor) -> torch.Tensor:
        """Get overlapped speech detection targets

        Parameters
        ----------
        collated_y : (batch_size, num_frames, num_speakers) tensor
            One-hot-encoding of current chunk speaker activity:
                * collated_y[b, f, s] = 1 if sth speaker is active at fth frame
                * collated_y[b, f, s] = 0 otherwise.

        Returns
        -------
        y : (batch_size, num_frames, ) np.ndarray
            y[b, f] = 1 if there is two or more active speakers at fth frame, 0 otherwise.
        """

        return 1 * (torch.sum(collated_y, dim=2, keepdim=False) > 1)
