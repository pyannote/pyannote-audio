#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

import collections
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.audio.labeling.models import StackedRNN
from pyannote.audio.util import from_numpy
from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.audio.labeling.extraction import SequenceLabeling


class ResegmentationGenerator(LabelingTaskGenerator):

    def __init__(self, precomputed, **kwargs):
        super(ResegmentationGenerator, self).__init__(
            precomputed, exhaustive=True, shuffle=True, **kwargs)

    def postprocess_y(self, Y):
        """Generate labels for resegmentation

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by `pyannote.audio.util.to_numpy`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.audio.util.to_numpy`
        """

        # +1 because...
        y = np.argmax(Y, axis=1) + 1

        # ... 0 is for non-speech
        non_speech = np.sum(Y, axis=1) == 0
        y[non_speech] = 0

        return np.int64(y)[:, np.newaxis]


class Resegmentation(LabelingTask):
    """

    Parameters
    ----------
    epochs : int, optional
        (Self-)train for that many epochs. Defaults to 10.
    ensemble : int, optional

    """

    # TODO -- ensemble of last K epochs

    def __init__(self, precomputed, epochs=10, ensemble=1, rnn='LSTM',
                 recurrent=[16, ], bidirectional=True, linear=[16, ], **kwargs):
        super(Resegmentation, self).__init__(**kwargs)
        self.precomputed = precomputed
        self.epochs = epochs
        self.ensemble = ensemble

        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.linear = linear

    def get_batch_generator(self, precomputed):
        return ResegmentationGenerator(
            precomputed, duration=self.duration, per_epoch=self.per_epoch,
            batch_size=self.batch_size, parallel=self.parallel)

    @property
    def n_classes(self):
        if not hasattr(self, 'n_classes_'):
            raise AttributeError('Call .apply() to set `n_classes` attribute')
        return self.n_classes_

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

    def _score(self, model, current_file, gpu=False):
        """Apply current model on current file

        Parameters
        ----------
        model : nn.Module
            Current state of the model.
        current_file : pyannote.database dict
            Current file.
        gpu : bool, optional
            Process on GPU. Defaults to False.

        Returns
        -------
        scores : SlidingWindowFeature
            Sequence labeling scores.
        """

        # initialize sequence labeling with model and features
        sequence_labeling = SequenceLabeling(
            model, self.precomputed, self.duration,
            step=.25*self.duration, batch_size=self.batch_size,
            source='audio', gpu=gpu)

        return sequence_labeling.apply(current_file)

    def _decode(self, scores):
        """Decoding

        Parameters
        ----------
        scores : iterable of SlidingWindowFeature instances
            Raw scores.

        Returns
        -------
        hypothesis : pyannote.core.Annotation
            Decoded scores.
        """

        # get ensemble scores (average of last self.ensemble epochs)
        avg_scores = sum(s.data for s in scores) / len(scores)

        # TODO. replace argmax by Viterbi decoding
        self.y_ = np.argmax(avg_scores, axis=1)
        return from_numpy(self.y_, self.precomputed,
                          labels=self.batch_generator_.labels)

    def apply_iter(self, current_file, hypothesis,
                   partial=True, gpu=False,
                   log_dir=None):
        """Yield re-segmentation results for each epoch

        Parameters
        ----------
        current_file : pyannote.database dict
            Currently processed file
        hypothesis : pyannote.core.Annotation
            Input segmentation
        partial : bool, optional
            Set to False to only yield final re-segmentation.
            Set to True to yield re-segmentation after each epoch.
        gpu : bool, optional
            Process on GPU.
        log_dir : str, optional
            Path to log directory.

        Yields
        ------
        resegmented : pyannote.core.Annotation
            Resegmentation results after each epoch.
        """

        current_file = dict(current_file)
        current_file['annotation'] = hypothesis

        # set `per_epoch` attribute to current file annotated duration
        self.per_epoch = get_annotated(current_file).duration()

        # number of speakers + 1 for non-speech
        self.n_classes_ = len(hypothesis.labels()) + 1

        model = StackedRNN(self.precomputed.dimension(), self.n_classes,
                           rnn=self.rnn, recurrent=self.recurrent,
                           linear=self.linear,
                           bidirectional=self.bidirectional,
                           logsoftmax=True)

        # initialize dummy protocol that has only one file
        protocol = self.get_dummy_protocol(current_file)

        if log_dir is not None:
            uri = get_unique_identifier(current_file)
            log_dir = 'f{log_dir}/{uri}'

        scores = collections.deque([], maxlen=self.ensemble)

        iterations = self.fit_iter(model, self.precomputed, protocol,
                                   log_dir=log_dir, epochs=self.epochs,
                                   gpu=gpu)

        for i, iteration in enumerate(iterations):

            # if 'partial', compute scores for every iteration
            # if not, compute scores for last 'ensemble' iterations only
            if partial or (i + 1 > self.epochs - self.ensemble):
                iteration_score = self._score(iteration['model'],
                                              current_file, gpu=gpu)
                scores.append(iteration_score)

            # if 'partial', generate (and yield) hypothesis
            if partial:
                hypothesis = self._decode(scores)
                yield hypothesis

        # generate (and yield) hypothesis in case it's not already
        if not partial:
            hypothesis = self._decode(scores)
            yield hypothesis

    def apply(self, current_file, hypothesis, gpu=False, log_dir=None):
        for hypothesis in self.apply_iter(current_file, hypothesis,
                                          partial=False, gpu=gpu):
            pass

        return hypothesis

