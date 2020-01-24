#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

raise NotImplementedError(
    'Resegmentation is broken for now. Come back later ;)')

import torch
import tempfile
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from pyannote.core import SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.audio.augmentation import AddNoiseFromGaps
from pyannote.core.utils.numpy import one_hot_decoding
from pyannote.audio.train.schedulers import ConstantScheduler
from torch.optim import SGD
from pyannote.audio.utils.path import mkdir_p
from pathlib import Path
from pyannote.audio.utils.signal import Binarize



class ResegmentationGenerator(LabelingTaskGenerator):
    """Batch generator for resegmentation self-training

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    current_file : `dict`

    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    mask_dimension : `int`, optional
        When set, batches will have a "mask" key that provides a mask that has
        the same length as "y". This "mask" will be passed to the loss function
        has a way to weigh samples according to their "mask" value. The actual
        value of `mask_dimension` is used to select which dimension to use.
        This option assumes that `current_file["mask"]` contains a
        `SlidingWindowFeature` that can be used as masking. Defaults to not use
        masking.
    mask_logscale : `bool`, optional
        Set to True to indicate that mask values are log scaled. Will apply
        exponential. Defaults to False. Has not effect when `mask_dimension`
        is not set.
    augmentation : `bool`, optional
        Augment (self-)training data by adding noise from non-speech regions.
        Defaults to False.
    """

    def __init__(self,
                 feature_extraction,
                 current_file,
                 resolution=None,
                 alignment=None,
                 duration=3.2,
                 batch_size=32,
                 mask_dimension=None,
                 mask_logscale=False,
                 augmentation=False):

        if 'duration' not in current_file:
            msg = (
                '`current_file` is expected to contain a "duration" key that '
                'provides the audio file duration in seconds.'
            )
            raise ValueError(msg)

        self.current_file = current_file
        dummy_protocol = self.get_dummy_protocol(current_file)

        self.augmentation = augmentation
        if augmentation:
            feature_extraction.augmentation = \
                AddNoiseFromGaps(protocol=dummy_protocol,
                                 subset='train',
                                 snr_min=10,
                                 snr_max=20)

        super(ResegmentationGenerator, self).__init__(
            feature_extraction,
            dummy_protocol,
            subset='train',
            resolution=resolution,
            alignment=alignment,
            duration=duration,
            step=0.25 * duration,
            batch_size=batch_size,
            exhaustive=True,
            shuffle=True,
            parallel=0 if self.augmentation else 1,
            mask_dimension=mask_dimension,
            mask_logscale=mask_logscale)

    def get_dummy_protocol(self, current_file):
        """Get dummy protocol containing only `current_file`

        Parameters
        ----------
        current_file : pyannote.database dict

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

    def postprocess_y(self, Y):
        """Generate labels for resegmentation

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by
            `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray
            y[t] = 0 indicates non-speech,
            y[t] = i + 1 indicates speaker i.

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        # +1 because...
        y = np.argmax(Y, axis=1) + 1

        # ... 0 is for non-speech
        non_speech = np.sum(Y, axis=1) == 0
        y[non_speech] = 0

        return np.int64(y)[:, np.newaxis]

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                         output=TaskOutput.SEQUENCE),
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': ['non_speech'] + self.segment_labels_},
        }

        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs

    @property
    def batches_per_epoch(self):
        """Number of batches needed to cover the whole file"""
        duration_per_epoch = 4 * self.current_file['duration']
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


class Resegmentation(LabelingTask):
    """Re-segmentation

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    Architecture :
    architecture_params :
    keep_sad: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    epochs : `int`, optional
        (Self-)train for that many epochs. Defaults to 30.
    ensemble : `int`, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    duration : `float`, optional
    batch_size : `int`, optional
    device : `torch.device`, optional
    mask_dimension : `int`, optional
        When set, batches will have a "mask" key that provides a mask that has
        the same length as "y". This "mask" will be passed to the loss function
        has a way to weigh samples according to their "mask" value. The actual
        value of `mask_dimension` is used to select which dimension to use.
        This option assumes that `current_file["mask"]` contains a
        `SlidingWindowFeature` that can be used as masking. Defaults to not use
        masking.
    mask_logscale : `bool`, optional
        Set to True to indicate that mask values are log scaled. Will apply
        exponential. Defaults to False. Has not effect when `mask_dimension`
        is not set.
    augmentation : `bool`, optional
        Augment (self-)training data by adding noise from non-speech regions.
        Defaults to False.
    """

    def __init__(self,
                 feature_extraction,
                 Architecture,
                 architecture_params,
                 keep_sad=False,
                 epochs=30,
                 learning_rate=0.1,
                 ensemble=1,
                 device=None,
                 duration=3.2,
                 batch_size=32,
                 mask_dimension=None,
                 mask_logscale=False,
                 augmentation=False):

        self.feature_extraction = feature_extraction
        self.Architecture = Architecture
        self.architecture_params = architecture_params
        self.keep_sad = keep_sad
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ensemble = ensemble

        self.mask_dimension = mask_dimension
        self.mask_logscale = mask_logscale

        self.augmentation = augmentation

        self.device = torch.device('cpu') if device is None else device

        super().__init__(duration=duration, batch_size=batch_size)

    def get_batch_generator(self, current_file):
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

        return ResegmentationGenerator(
            self.feature_extraction,
            current_file,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            batch_size=self.batch_size,
            mask_dimension=self.mask_dimension,
            mask_logscale=self.mask_logscale,
            augmentation=self.augmentation)

    def _decode(self, current_file, hypothesis, scores, labels):

        N, K = scores.data.shape

        if self.keep_sad:

            # K = 1 <~~> only non-speech
            # K = 2 <~~> just one speaker
            if K < 3:
                return hypothesis

            # sequence of most likely speaker index
            # (even when non-speech is the most likely class)
            best_speaker_indices = np.argmax(scores.data[:, 1:], axis=1) + 1

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                best_speaker_indices, scores.sliding_window, labels=labels)

            # revert non-speech regions back to original
            speech = hypothesis.get_timeline().support()
            new_hypothesis = new_hypothesis.crop(speech)

        else:

            # K = 1 <~~> only non-speech
            if K < 2:
                return hypothesis

            # sequence of most likely class index (including 0=non-speech)
            best_class_indices = np.argmax(scores.data, axis=1)

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                best_class_indices, scores.sliding_window, labels=labels)

        new_hypothesis.uri = hypothesis.uri
        return new_hypothesis

    def apply(self, current_file, hypothesis=None):
        """Apply resegmentation using self-supervised sequence labeling

        Parameters
        ----------
        current_file : `dict`
            Dictionary obtained by iterating over a subset of a
            `pyannote.database.Protocol` instance.
        hypothesis : `pyannote.core.Annotation`, optional
            Current diarization output. Defaults to current_file['hypothesis'].

        Returns
        -------
        new_hypothesis : `pyannote.core.Annotation`
            Updated diarization output.
        """

        # use current diarization output as "training" labels
        if hypothesis is None:
            hypothesis = current_file['hypothesis']
        current_file = dict(current_file)
        current_file['annotation'] = hypothesis

        batch_generator = self.get_batch_generator(current_file)

        model = self.Architecture(batch_generator.specifications,
                                  **self.architecture_params)

        # create a temporary directory to store models and log files
        # it is removed automatically before returning.
        with tempfile.TemporaryDirectory() as train_dir:

            # create train_dir/weights
            mkdir_p(Path(train_dir) / 'weights')

            epochs = self.fit_iter(
                model, batch_generator,
                warm_start=0, epochs=self.epochs,
                get_optimizer=SGD,
                scheduler=ConstantScheduler(),
                learning_rate=self.learning_rate,
                train_dir=Path(train_dir), verbosity=0,
                device=self.device)

            scores = []
            for i, current_model in enumerate(epochs):

                # do not compute scores that are not used in later ensembling
                # simply jump to next training epoch
                if i < self.epochs - self.ensemble:
                    continue

                current_model.eval()
                augmentation = self.feature_extraction.augmentation
                self.feature_extraction.augmentation = None

                # initialize sequence labeling with model and features
                sequence_labeling = SequenceLabeling(
                    model=current_model,
                    feature_extraction=self.feature_extraction,
                    duration=self.duration, step=.25 * self.duration,
                    batch_size=self.batch_size, device=self.device)

                # compute scores and keep track of them for later ensembling
                scores.append(sequence_labeling(current_file))

                current_model.train()
                self.feature_extraction.augmentation = augmentation

        # ensemble scores
        scores = SlidingWindowFeature(
            np.mean([s.data for s in scores], axis=0),
            scores[-1].sliding_window)

        # speaker labels
        labels = batch_generator.specifications['y']['classes'][1:]

        return self._decode(current_file, hypothesis, scores, labels)


class ResegmentationWithOverlap(Resegmentation):
    """Re-segmentation with overlap

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    Architecture :
    architecture_params :
    overlap_threshold : `float`, optional
        Defaults to 0.5.
    keep_sad: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    epochs : `int`, optional
        (Self-)train for that many epochs. Defaults to 30.
    ensemble : `int`, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    duration : `float`, optional
    batch_size : `int`, optional
    device : `torch.device`, optional
    mask_dimension : `int`, optional
        When set, batches will have a "mask" key that provides a mask that has
        the same length as "y". This "mask" will be passed to the loss function
        has a way to weigh samples according to their "mask" value. The actual
        value of `mask_dimension` is used to select which dimension to use.
        This option assumes that `current_file["mask"]` contains a
        `SlidingWindowFeature` that can be used as masking. Defaults to not use
        masking.
    mask_logscale : `bool`, optional
        Set to True to indicate that mask values are log scaled. Will apply
        exponential. Defaults to False. Has not effect when `mask_dimension`
        is not set.
    augmentation : `bool`, optional
        Augment (self-)training data by adding noise from non-speech regions.
        Defaults to False.
    """

    def __init__(self,
                 feature_extraction,
                 Architecture,
                 architecture_params,
                 keep_sad=False,
                 overlap_threshold=0.5,
                 epochs=30,
                 learning_rate=0.1,
                 ensemble=1,
                 device=None,
                 duration=3.2,
                 batch_size=32,
                 mask_dimension=None,
                 mask_logscale=False,
                 augmentation=False):

        super().__init__(feature_extraction,
                         Architecture,
                         architecture_params,
                         keep_sad=keep_sad,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         ensemble=ensemble,
                         device=device,
                         duration=duration,
                         batch_size=batch_size,
                         mask_dimension=mask_dimension,
                         mask_logscale=mask_logscale,
                         augmentation=augmentation)

        self.overlap_threshold = overlap_threshold
        self.binarizer_ = Binarize(onset=self.overlap_threshold,
                                   offset=self.overlap_threshold,
                                   scale='absolute',
                                   log_scale=True)

    def _decode(self, current_file, hypothesis, scores, labels):

        # obtain overlapped speech regions
        overlap = self.binarizer_.apply(current_file['overlap'], dimension=1)

        frames = scores.sliding_window
        N, K = scores.data.shape

        if self.keep_sad:

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
