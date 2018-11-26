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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.audio.train.trainer import Trainer


class Classifier(nn.Module):
    """MLP classifier

    Parameters
    ----------
    n_dimensions : int
        Embedding dimension
    n_classes : int
        Number of classes.
    """

    def __init__(self, n_dimensions, n_classes):
        super().__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

        self.hidden = nn.Linear(n_dimensions, n_dimensions, bias=True)
        self.output = nn.Linear(n_dimensions, n_classes, bias=True)

        self.logsoftmax_ = nn.LogSoftmax(dim=-1)

    def forward(self, embedding):
        hidden = F.tanh(self.hidden(embedding))
        return self.logsoftmax_(self.output(hidden))


class Softmax(Trainer):
    """Train embeddings in a supervised (classification) manner

    Parameters
    ----------
    duration : float, optional
        Use fixed duration segments with this `duration`.
        Defaults (None) to using variable duration segments.
    min_duration : float, optional
        In case `duration` is None, set segment minimum duration.
    max_duration : float, optional
        In case `duration` is None, set segment maximum duration.
    per_fold : int, optional
        Number of speakers per batch. Defaults to the whole speaker set.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 1.
    per_epoch : float, optional
        Number of days per epoch. Defaults to 7 (a week).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    CLASSES_TXT = '{log_dir}/classes.txt'
    CLASSIFIER_PT = '{log_dir}/weights/{epoch:04d}.classifier.pt'

    def __init__(self, duration=None, min_duration=None, max_duration=None,
                 per_label=1, per_fold=None, per_epoch=7, parallel=1,
                 label_min_duration=0.):
        super().__init__()

        self.per_fold = per_fold
        self.per_label = per_label
        self.per_epoch = per_epoch
        self.label_min_duration = label_min_duration

        self.duration = duration
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.parallel = parallel

        self.loss_ = nn.NLLLoss()

    def get_batch_generator(self, feature_extraction):
        """Get batch generator

        Parameters
        ----------
        feature_extraction : `pyannote.audio.features.FeatureExtraction`

        Returns
        -------
        generator : `pyannote.audio.embedding.generators.SpeechSegmentGenerator`
        """
        return SpeechSegmentGenerator(
            feature_extraction, label_min_duration=self.label_min_duration,
            per_label=self.per_label, per_fold=self.per_fold,
            per_epoch=self.per_epoch, duration=self.duration,
            min_duration=self.min_duration,
            max_duration=self.max_duration, parallel=self.parallel)

    def extra_init(self, model, device, checkpoint=None,
                   labels=None):
        """Initialize final classifier layers

        Parameters
        ----------
        model : `torch.nn.Module`
            Embedding model.
        device : `torch.device`
            Device used by model parameters.
        labels : `list` of `str`, optional
            List of classes.
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`, optional
            Checkpoint.

        Returns
        -------
        parameters : list
            Classifier parameters
        """

        # dimension of embedding space
        n_dimensions = model.output_dim

        # number of labels in training set
        n_classes = len(labels)

        self.classifier_ = Classifier(n_dimensions, n_classes)
        self.classifier_ = self.classifier_.to(device)

        # TODO. make sure classes_txt does not exist already
        # or, if it does, that it is coherent with "labels"
        log_dir = checkpoint.log_dir
        classes_txt = self.CLASSES_TXT.format(log_dir=log_dir)
        with open(classes_txt, mode='w') as fp:
            for label in labels:
                fp.write(f'{label}\n')

        return self.classifier_.parameters()

    def extra_restart(self, checkpoint, restart):
        """Load classifier weights

        Parameters
        ----------
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`
            Checkpoint.
        restart : `int`
            Epoch used for warm restart.
        """

        classifier_pt = self.CLASSIFIER_PT.format(
            log_dir=checkpoint.log_dir, epoch=restart)
        classifier_state = torch.load(classifier_pt,
            map_location=lambda storage, loc: storage)
        self.classifier_.load_state_dict(classifier_state)

    def batch_loss(self, batch, model, device, writer=None, **kwargs):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
        model : `torch.nn.Module`
            Model currently being trained.
        device : `torch.device`
            Device used by model parameters.
        writer : `tensorboardX.SummaryWriter`, optional
            Tensorboard writer.

        Returns
        -------
        loss : `torch.Tensor`
            Negative log likelihood loss
        """

        fX = self.forward(batch, model, device)
        y_pred = self.classifier_(fX)
        y = torch.tensor(np.stack(batch['y'])).to(device)
        return self.loss_(y_pred, y)

    def on_epoch_end(self, iteration, checkpoint, **kwargs):
        """Save classifier to disk at the end of current epoch

        Parameters
        ----------
        iteration : `int`
            Current epoch.
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`
            Checkpoint.
        """

        classifier_pt = self.CLASSIFIER_PT.format(
            log_dir=checkpoint.log_dir, epoch=iteration)
        torch.save(self.classifier_.state_dict(), classifier_pt)
