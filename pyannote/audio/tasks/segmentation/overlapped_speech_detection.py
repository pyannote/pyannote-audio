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


from typing import Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np
from pyannote.core import Segment, SlidingWindowFeature
from pyannote.database import Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.tasks.segmentation.mixins import SegmentationTask


class OverlappedSpeechDetection(SegmentationTask):
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
    cache : str, optional
        As (meta-)data preparation might take a very long time for large datasets,
        it can be cached to disk for later (and faster!) re-use.
        When `cache` does not exist, `Task.prepare_data()` generates training
        and validation metadata from `protocol` and save them to disk.
        When `cache` exists, `Task.prepare_data()` is skipped and (meta)-data
        are loaded from disk. Defaults to a temporary path.
    duration : float, optional
        Chunks duration. Defaults to 2s.
    warm_up : float or (float, float), optional
        Use that many seconds on the left- and rightmost parts of each chunk
        to warm up the model. While the model does process those left- and right-most
        parts, only the remaining central part of each chunk is used for computing the
        loss during training, and for aggregating scores during inference.
        Defaults to 0. (i.e. no warm-up).
    balance: Sequence[Text], optional
        When provided, training samples are sampled uniformly with respect to these keys.
        For instance, setting `balance` to ["database","subset"] will make sure that each
        database & subset combination will be equally represented in the training samples.
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
        balance: Optional[Sequence[Text]] = None,
        weight: Optional[Text] = None,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        augmentation: Optional[BaseWaveformTransform] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        cache: Optional[Union[str, None]] = None,
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
            cache=cache,
        )

        self.specifications = Specifications(
            problem=Problem.BINARY_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[
                "overlap",
            ],
        )

        self.overlap = overlap
        self.balance = balance
        self.weight = weight

    def prepare_chunk(self, file_id: int, start_time: float, duration: float):
        """Prepare chunk for overlapped speech detection

        Parameters
        ----------
        file_id : int
            File index
        start_time : float
            Chunk start time
        duration : float
            Chunk duration.

        Returns
        -------
        sample : dict
            Dictionary containing the chunk data with the following keys:
            - `X`: waveform
            - `y`: target as a SlidingWindowFeature instance
            - `meta`:
                - `database`: database index
                - `file`: file index
        """

        file = self.get_file(file_id)

        chunk = Segment(start_time, start_time + duration)

        sample = dict()
        sample["X"], _ = self.model.audio.crop(file, chunk, duration=duration)

        # gather all annotations of current file
        annotations = self.prepared_data["annotations-segments"][
            self.prepared_data["annotations-segments"]["file_id"] == file_id
        ]

        # gather all annotations with non-empty intersection with current chunk
        chunk_annotations = annotations[
            (annotations["start"] < chunk.end) & (annotations["end"] > chunk.start)
        ]

        # discretize chunk annotations at model output resolution
        step = self.model.receptive_field.step
        half = 0.5 * self.model.receptive_field.duration

        start = np.maximum(chunk_annotations["start"], chunk.start) - chunk.start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunk_annotations["end"], chunk.end) - chunk.start - half
        end_idx = np.round(end / step).astype(int)

        # frame-level targets
        num_frames = self.model.num_frames(
            round(duration * self.model.hparams.sample_rate)
        )
        y = np.zeros((num_frames, 1), dtype=np.uint8)
        for start, end in zip(start_idx, end_idx):
            y[start : end + 1, 0] += 1
        y = 1 * (y > 1)

        sample["y"] = SlidingWindowFeature(
            y, self.model.receptive_field, labels=["speech"]
        )

        metadata = self.prepared_data["audio-metadata"][file_id]
        sample["meta"] = {key: metadata[key] for key in metadata.dtype.names}
        sample["meta"]["file"] = file_id

        return sample
