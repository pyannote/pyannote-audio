#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

"""Resegmentation"""

from typing import Optional
from typing import Type
from typing import Iterable
from typing import Dict
from typing import Text
import scipy.signal

import torch
import tempfile
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database import get_annotated
from pyannote.core.utils.numpy import one_hot_decoding
from pyannote.core.utils.numpy import one_hot_encoding
from pyannote.audio.train.schedulers import ConstantScheduler
from torch.optim import SGD
from pathlib import Path
from pyannote.audio.utils.signal import Binarize

from pyannote.audio.features import FeatureExtraction
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import Alignment


class ResegmentationGenerator(LabelingTaskGenerator):
    """Batch generator for resegmentation self-training

    Parameters
    ----------
    current_file : `ProtocolFile`
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of audio chunks. Defaults to 4s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    allow_overlap : bool, optional
        Allow overlapping speakers. Defaults to False.
        Does not really work for now...
    mask : str, optional
        When provided, current_file[mask] is used by the loss function to weigh
        samples.
    """

    def __init__(self,
                 current_file: ProtocolFile,
                 resolution: Optional[Resolution] = None,
                 alignment: Optional[Alignment] = None,
                 duration: float = 2,
                 batch_size: int = 32,
                 lock_speech: bool = False,
                 allow_overlap: bool = False,
                 mask: Text = None):

        self.current_file = current_file
        self.allow_overlap = allow_overlap
        self.lock_speech = lock_speech

        super().__init__('@features',
                         self.get_dummy_protocol(current_file),
                         subset='train',
                         resolution=resolution,
                         alignment=alignment,
                         duration=duration,
                         batch_size=batch_size,
                         mask=mask)

    @property
    def resolution(self):
        return self.current_file['features'].sliding_window

    def get_dummy_protocol(self, current_file: ProtocolFile) -> SpeakerDiarizationProtocol:
        """Get dummy protocol containing only `current_file`

        Parameters
        ----------
        current_file : ProtocolFile

        Returns
        -------
        protocol : SpeakerDiarizationProtocol instance
            Dummy protocol containing only `current_file` in both train,
            dev., and test sets.

        """

        class DummyProtocol(SpeakerDiarizationProtocol):

            def trn_iter(self):
                yield current_file

            def dev_iter(self):
                yield current_file

            def tst_iter(self):
                yield current_file

        return DummyProtocol()

    def postprocess_y(self, Y: np.ndarray) -> np.ndarray:
        """Generate labels for resegmentation

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by
            `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) or (n_samples, n_speakers) numpy.ndarray
            When allow_overlap is True, y has shape (n_samples, n_speakers)
                * y[t, i] = 1 means speaker i is active
            When allow_overlap is False, y has shape (n_samples, 1)
                * y[t] = 0 indicates non-speech,
                * y[t] = i + 1 indicates speaker i.

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        # when allowing overlap, multiple speakers can be active at once.
        # hence, Y sticks to shape (n_samples, n_speakers)
        if self.allow_overlap:
            return Y

        # when overlap is not allowed, reshape Y to (n_samples, 1)
        else:

            # if speech / non-speech status is locked
            # there is no class for non-speech.
            if self.lock_speech:
                y = np.argmax(Y, axis=1)
                return np.int64(y)[:, np.newaxis]

            # otherwise, we add a non-speech class at index 0
            else:

                # +1 because...
                y = np.argmax(Y, axis=1) + 1

                # ... 0 is for non-speech
                non_speech = np.sum(Y, axis=1) == 0
                y[non_speech] = 0

                return np.int64(y)[:, np.newaxis]

    @property
    def specifications(self) -> Dict:
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {'X': {'dimension': self.current_file['features'].dimension}}

        if self.allow_overlap:
            # when allowing overlap, multiple speakers can be active at the
            # same time (hence multi-label classification task)
            specs['task'] = Task(type=TaskType.MULTI_LABEL_CLASSIFICATION,
                                 output=TaskOutput.SEQUENCE)
            specs['y'] = {'classes': self.segment_labels_}

        else:
            # when overlap is not allowed, only one speaker can be active at
            # a particular time (hence: multi-class classification)
            specs['task'] = Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                                 output=TaskOutput.SEQUENCE)

            if self.lock_speech:
                # when locking speech / non-speech status,
                # there is no non-speech class
                specs['y'] = {'classes': self.segment_labels_}
            else:
                # when speech / non-speech status can be updated,
                # one must add a non-speech class
                specs['y'] = {'classes': ['non_speech'] + self.segment_labels_}

        # for key, classes in self.file_labels_.items():
        #     specs[key] = {'classes': classes}

        return specs


class Resegmentation(LabelingTask):
    """Re-segmentation

    Parameters
    ----------
    feature_extraction : FeatureExtraction
        Feature extraction.
    Architecture : Model subclass
    architecture_params : dict
    epochs : `int`, optional
        (Self-)train for that many epochs. Defaults to 5.
    ensemble : `int`, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1.
    batch_size : int, optional
        Batch size. Defaults to 32.
    device : `torch.device`, optional
    lock_speech: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    allow_overlap : bool, optional
        Allow overlapping speakers. Defaults to False.
    mask : str, optional
        When provided, current_file[mask] is used by the loss function to weigh
        samples.
    """

    def __init__(self, feature_extraction: FeatureExtraction,
                 Architecture: Type[Model],
                 architecture_params: dict,
                 lock_speech: bool = False,
                 epochs: int = 5,
                 learning_rate: float = 0.1,
                 ensemble: int = 1,
                 duration: float = 2.0,
                 step: float = 0.1,
                 n_jobs: int = 1,
                 device: torch.device = None,
                 batch_size: int = 32,
                 allow_overlap: bool = False,
                 mask: Text = None):

        self.feature_extraction = feature_extraction

        self.Architecture = Architecture
        self.architecture_params = architecture_params

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.ensemble = ensemble

        self.n_jobs = n_jobs

        self.lock_speech = lock_speech
        self.allow_overlap = allow_overlap
        self.mask = mask

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_ = torch.device(device)

        super().__init__(duration=duration,
                         batch_size=batch_size,
                         per_epoch=None,
                         step=step)

    def get_batch_generator(self, current_file: ProtocolFile) -> ResegmentationGenerator:
        """Get batch generator for current file

        Parameters
        ----------
        current_file : `dict`
            Dictionary obtained by iterating over a subset of a
            `pyannote.database.Protocol` instance.

        Returns
        -------
        batch_generator : `ResegmentationGenerator`
        """

        resolution = self.Architecture.get_resolution(
            **self.architecture_params)
        alignment = self.Architecture.get_alignment(
            **self.architecture_params)

        return ResegmentationGenerator(current_file,
                                       resolution=resolution,
                                       alignment=alignment,
                                       duration=self.duration,
                                       batch_size=self.batch_size,
                                       lock_speech=self.lock_speech,
                                       allow_overlap=self.allow_overlap,
                                       mask=self.mask)

    def _decode(self, current_file: ProtocolFile,
                      hypothesis: Annotation,
                      scores: SlidingWindowFeature,
                      labels: Iterable) -> Annotation:

        N, K = scores.data.shape

        if self.allow_overlap:
            active_speakers = scores.data > 0.5

        else:
            if self.lock_speech:
                active_speakers = np.argmax(scores.data, axis=1) + 1

            else:
                active_speakers = np.argmax(scores.data, axis=1)

        # reconstruct annotation
        new_hypothesis = one_hot_decoding(
            active_speakers, scores.sliding_window, labels=labels)

        new_hypothesis.uri = hypothesis.uri

        if self.lock_speech:
            speech = hypothesis.get_timeline().support()
            new_hypothesis = new_hypothesis.crop(speech)

        return new_hypothesis

    def __call__(self, current_file: ProtocolFile,
                       hypothesis: Annotation,
                       debugging: bool = False) -> Annotation:
        """Apply resegmentation using self-supervised sequence labeling

        Parameters
        ----------
        current_file : ProtocolFile
            Dictionary obtained by iterating over a subset of a
            `pyannote.database.Protocol` instance.
        hypothesis : Annotation, optional
            Current diarization output. Defaults to current_file['hypothesis'].

        Returns
        -------
        new_hypothesis : Annotation
            Updated diarization output.
        """

        # make sure current_file is not modified
        current_file = dict(current_file)

        current_file['annotation'] = hypothesis
        current_file['features'] = self.feature_extraction(current_file)

        debug = {}

        # when locking speech / non-speech status, we add a (or update
        # existing) mask so that the loss is not computed on non-speech regions
        if self.lock_speech:

            encoded, _ = one_hot_encoding(hypothesis,
                                          get_annotated(current_file),
                                          current_file['features'].sliding_window,
                                          mode='center')
            speech = 1. * (np.sum(encoded, axis=1, keepdims=True) > 0)
            current_file['speech'] = speech
            debug['speech'] = speech

            if self.mask is None:
                self.mask = 'speech'

            else:
                mask = current_file[self.mask]
                current_file[self.mask] = mask * speech.align(mask)

            debug['mask'] = current_file[self.mask]

        batch_generator = self.get_batch_generator(current_file)

        model = self.Architecture(batch_generator.specifications,
                                  **self.architecture_params)

        chunks = SlidingWindow(duration=self.duration,
                               step=self.step * self.duration)

        # create a temporary directory to store models and log files
        # it is removed automatically before returning.
        with tempfile.TemporaryDirectory() as train_dir:

            epochs = self.fit_iter(model,
                                   batch_generator,
                                   warm_start=0,
                                   epochs=self.epochs,
                                   get_optimizer=SGD,
                                   scheduler=ConstantScheduler(),
                                   learning_rate=self.learning_rate,
                                   train_dir=Path(train_dir),
                                   verbosity=1,
                                   device=self.device,
                                   callbacks=None,
                                   n_jobs=self.n_jobs)

            scores = []
            for i, current_model in enumerate(epochs):

                # do not compute scores that are not used in later ensembling
                # simply jump to next training epoch (except when debugging)
                if not debugging and i < self.epochs - self.ensemble:
                    continue

                current_model.eval()

                scores.append(current_model.slide(
                    current_file['features'],
                    chunks,
                    batch_size=self.batch_size,
                    device=self.device,
                    return_intermediate=None,
                    progress_hook=None))
                current_model.train()

        debug['scores'] = scores

        # ensemble scores
        scores = SlidingWindowFeature(
            np.mean([s.data for s in scores[-self.ensemble:]], axis=0),
            scores[-1].sliding_window)
        debug['final_scores'] = scores

        labels = batch_generator.specifications['y']['classes']
        if not self.lock_speech:
            labels = labels[1:]

        debug['labels'] = labels

        decoded = self._decode(current_file,
                               hypothesis,
                               scores,
                               labels)

        decoded.debug = debug
        return decoded


class ResegmentationWithOverlap(Resegmentation):
    """Re-segmentation with overlap

    Parameters
    ----------
    feature_extraction : FeatureExtraction
        Feature extraction.
    Architecture : Model subclass
    architecture_params : dict
    overlap_threshold : `float`, optional
        Defaults to 0.5.
    lock_speech: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    epochs : `int`, optional
        (Self-)train for that many epochs. Defaults to 5.
    ensemble : `int`, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1.
    batch_size : int, optional
        Batch size. Defaults to 32.
    device : `torch.device`, optional
    mask : str, optional
        When provided, current_file[mask] is used by the loss function to weigh
        samples.
    """

    def __init__(self,
                 feature_extraction: FeatureExtraction,
                 Architecture: Type[Model],
                 architecture_params: dict,
                 lock_speech: bool = False,
                 overlap_threshold: float = 0.5,
                 epochs: int = 5,
                 learning_rate: float = 0.1,
                 ensemble: int = 1,
                 duration: float = 2.0,
                 step: float = 0.1,
                 n_jobs: int = 1,
                 device: torch.device = None,
                 batch_size: int = 32,
                 mask: Text = None):

        super().__init__(feature_extraction,
                         Architecture,
                         architecture_params,
                         lock_speech=lock_speech,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         ensemble=ensemble,
                         duration=duration,
                         step=step,
                         n_jobs=n_jobs,
                         device=device,
                         batch_size=batch_size,
                         mask=mask)

        self.overlap_threshold = overlap_threshold
        self.binarizer_ = Binarize(onset=self.overlap_threshold,
                                   offset=self.overlap_threshold,
                                   scale='absolute',
                                   log_scale=True)

    def _decode(self, current_file: ProtocolFile,
                      hypothesis: Annotation,
                      scores: SlidingWindowFeature,
                      labels: Iterable) -> Annotation:

        # obtain overlapped speech regions
        overlap = self.binarizer_.apply(current_file['overlap'], dimension=1)

        frames = scores.sliding_window
        N, K = scores.data.shape

        if self.lock_speech:

            # K = 1 <~~> only non-speech
            # K = 2 <~~> just one speaker
            if K < 3:
                return hypothesis

            # sequence of two most likely speaker indices
            # (even when non-speech is in fact the most likely class)
            best_speakers_indices = np.argsort(
                -scores.data[:, 1:], axis=1)[:, :2]

            active_speakers = np.zeros((N, K-1), dtype=np.int64)

            # start by assigning most likely speaker...
            for t, k in enumerate(best_speakers_indices[:, 0]):
                active_speakers[t, k] = 1

            # ... then add second most likely speaker in overlap regions
            T = frames.crop(overlap, mode='strict')

            # because overlap may use a different feature extraction step
            # it might happen that T contains indices slightly large than
            # the actual number of frames. the line below remove any such
            # indices.
            T = T[T < N]

            # mark second most likely speaker as active
            active_speakers[T, best_speakers_indices[T, 1]] = 1

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                active_speakers, frames, labels=labels)

            # revert non-speech regions back to original
            speech = hypothesis.get_timeline().support()
            new_hypothesis = new_hypothesis.crop(speech)

        else:

            # K = 1 <~~> only non-speech
            if K < 2:
                return hypothesis

            # sequence of two most likely class indices
            # sequence of two most likely class indices
            # (including 0=non-speech)
            best_speakers_indices = np.argsort(
                -scores.data, axis=1)[:, :2]

            active_speakers = np.zeros((N, K-1), dtype=np.int64)

            # start by assigning the most likely speaker...
            for t, k in enumerate(best_speakers_indices[:, 0]):
                # k = 0 is for non-speech
                if k > 0:
                    active_speakers[t, k-1] = 1

            # ... then add second most likely speaker in overlap regions
            T = frames.crop(overlap, mode='strict')

            # because overlap may use a different feature extraction step
            # it might happen that T contains indices slightly large than
            # the actual number of frames. the line below remove any such
            # indices.
            T = T[T < N]

            # remove timesteps where second most likely class is non-speech
            T = T[best_speakers_indices[T, 1] > 0]

            # mark second most likely speaker as active
            active_speakers[T, best_speakers_indices[T, 1] - 1] = 1

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                active_speakers, frames, labels=labels)

        new_hypothesis.uri = hypothesis.uri
        return new_hypothesis
