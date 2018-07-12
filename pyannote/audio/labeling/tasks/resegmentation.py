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

import torch
import tempfile
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
from pyannote.audio.train.schedulers import ConstantScheduler
from torch.optim import SGD


class ResegmentationGenerator(LabelingTaskGenerator):
    """Batch generator for resegmentation self-training

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

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
    """Resegmentation based on stacked recurrent neural network

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    epochs : int, optional
        (Self-)train for that many epochs. Defaults to 10.
    ensemble : int, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    """

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

    def _score(self, model, current_file, device=None):
        """Apply current model on current file

        Parameters
        ----------
        model : nn.Module
            Current state of the model.
        current_file : pyannote.database dict
            Current file.
        device : torch.device

        Returns
        -------
        scores : SlidingWindowFeature
            Sequence labeling scores.
        """

        # initialize sequence labeling with model and features
        sequence_labeling = SequenceLabeling(
            model, self.precomputed, duration=self.duration,
            step=.25*self.duration, batch_size=self.batch_size,
            source='audio', device=device)

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
                   partial=True, device=None,
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
        device : torch.device, optional
            Defaults to torch.device('cpu')
        log_dir : str, optional
            Path to log directory.

        Yields
        ------
        resegmented : pyannote.core.Annotation
            Resegmentation results after each epoch.
        """

        device = torch.device('cpu') if device is None else device

        current_file = dict(current_file)
        current_file['annotation'] = hypothesis

        # set `per_epoch` attribute to current file annotated duration
        self.per_epoch = get_annotated(current_file).duration()

        # number of speakers + 1 for non-speech
        self.n_classes_ = len(hypothesis.labels()) + 1

        model = StackedRNN(self.precomputed.dimension, self.n_classes,
                           rnn=self.rnn, recurrent=self.recurrent,
                           linear=self.linear,
                           bidirectional=self.bidirectional,
                           logsoftmax=True)

        # initialize dummy protocol that has only one file
        protocol = self.get_dummy_protocol(current_file)

        if log_dir is None:
            log_dir = tempfile.mkdtemp()
        uri = get_unique_identifier(current_file)
        log_dir = 'f{log_dir}/{uri}'

        self.scores_ = collections.deque([], maxlen=self.ensemble)

        iterations = self.fit_iter(
            model, self.precomputed, protocol, subset='train',
            restart=0, epochs=self.epochs, learning_rate='auto',
            get_optimizer=SGD, get_scheduler=ConstantScheduler,
            log_dir=log_dir, device=device)

        for i, iteration in enumerate(iterations):

            # if 'partial', compute scores for every iteration
            # if not, compute scores for last 'ensemble' iterations only
            if partial or (i + 1 > self.epochs - self.ensemble):
                iteration_score = self._score(iteration['model'],
                                              current_file, device=device)
                self.scores_.append(iteration_score)

            # if 'partial', generate (and yield) hypothesis
            if partial:
                hypothesis = self._decode(self.scores_)
                yield hypothesis

        # generate (and yield) final hypothesis in case it's not already
        if not partial:
            hypothesis = self._decode(self.scores_)
            yield hypothesis

    def apply(self, current_file, hypothesis, device=None, log_dir=None):
        """Apply re-segmentation

        Parameters
        ----------
        current_file : pyannote.database dict
            Currently processed file
        hypothesis : pyannote.core.Annotation
            Input segmentation
        device : torch.device, optional
            Defaults to torch.device('cpu')
        log_dir : str, optional
            Path to log directory.

        Returns
        -------
        resegmented : pyannote.core.Annotation
            Resegmentation result
        """

        for hypothesis in self.apply_iter(current_file, hypothesis,
                                          partial=False, device=device):
            pass

        return hypothesis
