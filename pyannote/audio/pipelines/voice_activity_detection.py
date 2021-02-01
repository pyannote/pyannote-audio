# MIT License
#
# Copyright (c) 2018-2020 CNRS
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

"""Voice activity detection pipelines"""

import math
import warnings
from copy import deepcopy
from typing import List, Optional, Text, Tuple, Union

import numpy as np
import scipy.signal
from torch.utils.data import DataLoader, IterableDataset
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.task import Problem, Resolution, Specifications, Task
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Categorical, Uniform


class OracleVoiceActivityDetection(Pipeline):
    """Oracle voice activity detection pipeline"""

    def apply(self, file: AudioFile) -> Annotation:
        """Return groundtruth voice activity detection

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speech regions
        """

        speech = file["annotation"].get_timeline().support()
        return speech.to_annotation(generator="string", modality="speech")


class VoiceActivityDetection(Pipeline):
    """Voice activity detection pipeline

    Parameters
    ----------
    scores : Inference or str, optional
        `Inference` instance used to extract raw voice activity detection scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "vad".
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on, min_duration_off : float
        Minimum duration in either state (speech or not)

    """

    def __init__(self, scores: Union[Inference, Text] = "vad", fscore: bool = False):
        super().__init__()

        self.scores = scores
        self.fscore = fscore

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        #  hyper-parameters used for post-processing
        # i.e. removing short speech/non-speech regions
        self.min_duration_on = Uniform(0.0, 2.0)
        self.min_duration_off = Uniform(0.0, 2.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        speech : `pyannote.core.Annotation`
            Speech regions.
        """

        if isinstance(self.scores, Inference):
            speech_probability = self.scores(file)
        else:
            speech_probability = file[self.scores]

        speech = self._binarize(speech_probability)
        speech.uri = file.get("uri", None)
        return speech.to_annotation(generator="string", modality="speech")

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        if self.fscore:
            return DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False)

        return DetectionErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        if self.fscore:
            return "maximize"
        return "minimize"


class ValDataset(IterableDataset):
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.val__iter__()

    def __len__(self):
        return self.task.val__len__()


class _AdaptiveVoiceActivityDetectionTask(Task):

    ACRONYM = "vad"

    def __init__(
        self,
        file: AudioFile,
        duration: float = 2.0,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
    ):

        self.file = file

        super().__init__(
            None,
            duration=duration,
            min_duration=None,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
        )

        self.specifications = Specifications(
            problem=Problem.BINARY_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            classes=[
                "speech",
            ],
        )

    def prepare_y(self, one_hot_y: np.ndarray):
        """Get voice activity detection targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if at least one speaker is active at tth frame, 0 otherwise.
        """
        return np.int64(np.sum(one_hot_y, axis=1) > 0)

    @property
    def chunk_labels(self) -> Optional[List[Text]]:
        """Ordered list of labels

        Override this method to make `prepare_chunk` use a specific
        ordered list of labels when extracting frame-wise labels.

        See `prepare_chunk` source code for details.
        """
        return None

    def prepare_chunk(
        self,
        file: AudioFile,
        chunk: Segment,
        duration: float = None,
        weight: Text = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Text]]:
        """Extract audio chunk and corresponding frame-wise labels

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : Segment
            Audio chunk.
        duration : float, optional
            Fix chunk duration to avoid rounding errors. Defaults to self.duration
        weight : str, optional
            When provided, file[weight] is expected to be a SlidingWindowFeature instance.
            This will add a "w" keys to the returned dictionary that contains the output of
            file[weight].crop(chunk), interpolated such that it has the same number of frames
            as y.

        Returns
        -------
        sample : dict
            Dictionary with the following keys:
            X : np.ndarray
                Audio chunk as (num_samples, num_channels) array.
            y : np.ndarray
                Frame-wise labels as (num_frames, num_labels) array.
            labels : list of str
                Ordered labels such that y[:, k] corresponds to activity of labels[k].
            weight : np.ndarray, optional
                Frame-wise weights as (num_frames, ) array.
        """

        X, _ = self.model.audio.crop(
            file,
            chunk,
            mode="center",
            fixed=self.duration if duration is None else duration,
        )

        introspection = self.model.introspection

        if self.is_multi_task:
            # this assumes that all tasks share the same model introspection.
            # this is a reasonable assumption for now.
            any_task = next(iter(introspection.keys()))
            num_frames, _ = introspection[any_task](X.shape[1])
        else:
            num_frames, _ = introspection(X.shape[1])

        annotation = file["annotation"].crop(chunk)
        labels = annotation.labels() if self.chunk_labels is None else self.chunk_labels

        y = np.zeros((num_frames, len(labels)), dtype=np.int8)
        frames = SlidingWindow(
            start=chunk.start,
            duration=self.duration / num_frames,
            step=self.duration / num_frames,
        )
        for label in annotation.labels():
            try:
                k = labels.index(label)
            except ValueError:
                warnings.warn(
                    f"File {file['uri']} contains unexpected label '{label}'."
                )
                continue

            segments = annotation.label_timeline(label)
            for start, stop in frames.crop(segments, mode="center", return_ranges=True):
                y[start:stop, k] += 1

        # handle corner case when the same label is active more than once
        y = np.minimum(y, 1, out=y)

        sample = {"X": X, "y": y, "labels": labels}

        if weight is not None:
            sample["weight"] = (
                file[weight].crop(chunk, fixed=duration, mode="center").squeeze(axis=1)
            )

        return sample

    def train__iter__(self):

        # create epoch-specific random number generator
        rng = create_rng_for_worker(self.model.current_epoch)

        # deep copy file as we are going to update its "confidence" key
        file = deepcopy(self.file)

        # mask half of confidence (i.e. set to 0), using periodic square signal
        num_frames = file["confidence"].data.shape[0]
        num_frames_per_chunk = file["confidence"].sliding_window.duration_to_samples(
            self.duration
        )
        train_random_shift = rng.randint(0, num_frames_per_chunk)
        print(train_random_shift)
        train_mask = scipy.signal.square(
            np.pi
            / num_frames_per_chunk
            * np.arange(train_random_shift, num_frames + train_random_shift)
        )
        file["confidence"].data *= 0.5 * (1 + train_mask[:, np.newaxis])

        chunks = list()
        for segment in file["annotated"]:
            if segment.duration < self.duration:
                continue
            # slide a window on the whole "annotated" duration with 50% overlap
            chunks.extend(
                SlidingWindow(
                    # randomize sliding window start time so that
                    # we never see the exact same chunk twice
                    start=rng.uniform(
                        segment.start,
                        min(segment.end - self.duration, segment.start + self.duration),
                    ),
                    duration=self.duration,
                    step=0.5 * self.duration,
                    end=segment.end - self.duration,
                )
            )

        # shuffle chunks order in order to avoid training bias due to, e.g. speakers that
        # would mostly speaker at the beginning of the file
        rng.shuffle(chunks)

        # loop on shuffled chunks
        for chunk in chunks:
            sample = self.prepare_chunk(
                file, chunk, duration=self.duration, weight="confidence"
            )
            sample["y"] = self.prepare_y(sample["y"])
            _ = sample.pop("labels")
            yield sample

    def train__len__(self):
        # Number of training samples in one epoch
        return sum(
            max(0, math.floor(2 * segment.duration / self.duration - 3))
            for segment in self.file["annotated"]
            if segment.duration >= self.duration
        )

    def val__iter__(self):

        # create epoch-specific random number generator
        # (uses the same seed as corresponding train__iter__)
        rng = create_rng_for_worker(self.model.current_epoch)

        file = deepcopy(self.file)

        # mask half of confidence (i.e. set to 0), using periodic square signal
        num_frames = file["confidence"].data.shape[0]
        num_frames_per_chunk = file["confidence"].sliding_window.duration_to_samples(
            self.duration
        )

        # make sure train and validation masks do not overlap
        train_random_shift = rng.randint(0, num_frames_per_chunk)
        print(train_random_shift)
        val_random_shift = train_random_shift + num_frames_per_chunk
        val_mask = scipy.signal.square(
            np.pi
            / num_frames_per_chunk
            * np.arange(val_random_shift, num_frames + val_random_shift)
        )
        # TODO: should we use binary or soft confidence in validation?
        # file["confidence"].data = 0.5 * (1 + val_mask[:, np.newaxis])
        file["confidence"].data *= 0.5 * (1 + val_mask[:, np.newaxis])

        # slide a window on the whole "annotated" duration with 50% overlap
        for segment in file["annotated"]:

            if segment.duration < self.duration:
                continue

            for chunk in SlidingWindow(
                start=segment.start,
                duration=self.duration,
                step=0.5 * self.duration,
                end=segment.end - self.duration,
            ):

                sample = self.prepare_chunk(
                    file, chunk, duration=self.duration, weight="confidence"
                )
                sample["y"] = self.prepare_y(sample["y"])
                _ = sample.pop("labels")
                yield sample

    def val__len__(self):
        # Number of validation samples in one epoch
        return sum(
            max(0, math.floor(2 * segment.duration / self.duration - 1))
            for segment in self.file["annotated"]
            if segment.duration >= self.duration
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        return DataLoader(
            ValDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class AdaptiveVoiceActivityDetection(Pipeline):
    def __init__(self):
        super().__init__()

        # hyper-parameters
        self.batch_size = Categorical([1, 2, 4, 8, 16])

    def apply(self, file: AudioFile) -> Annotation:

        model: Model = Model.from_pretrained(
            "hbredin/VoiceActivityDetection-PyanNet-DIHARD"
        )
        inference = Inference(model)
        pipeline: VoiceActivityDetection = Pipeline.from_pretrained(
            "hbredin/VoiceActivityDetection-PyanNet-DIHARD"
        )

        pipeline.scores = "soft_vad"
        soft_vad = inference(file)
        file["soft_vad"] = soft_vad
        hard_vad = pipeline(file)

        confidence = deepcopy(soft_vad)
        confidence.data = np.abs(confidence.data - 0.5) / 0.5

        f = {
            "annotated": file["annotated"],
            "annotation": hard_vad,
            "confidence": confidence,
        }

        task = _AdaptiveVoiceActivityDetectionTask(
            f, duration=model.specifications.duration, batch_size=self.batch_size
        )
        model.task = task

        monitor, direction = task.val_monitor

        # early_stopping = EarlyStopping(
        #     monitor=monitor,
        #     mode=direction,
        #     min_delta=0.0,
        #     patience=100,
        #     strict=True,
        #     verbose=False,
        # )

        # callbacks = [early_stopping]

        # logger = TensorBoardLogger(
        #     ".",
        #     name="",
        #     version="",
        #     log_graph=False,  # TODO: fixes onnx error with asteroid-filterbanks
        # )

        # trainer = Trainer(callbacks=callbacks, logger=logger)
        # trainer.fit(model)
