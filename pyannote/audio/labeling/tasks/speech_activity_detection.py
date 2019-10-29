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

"""Speech activity detection"""

import numpy as np
import torch
import torch.nn as nn
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_CLASS_CLASSIFICATION
from ..gradient_reversal import GradientReversal
from pyannote.audio.models.models import RNN

class SpeechActivityDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training speech activity detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    frame_info : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    frame_crop : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def postprocess_y(self, Y):
        """Generate labels for speech activity detection

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by
            `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        # number of speakers for each frame
        speaker_count = np.sum(Y, axis=1, keepdims=True)

        # mark speech regions as such
        return np.int64(speaker_count > 0)

    @property
    def specifications(self):
        specs = {
            'task': TASK_MULTI_CLASS_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': ['non_speech', 'speech']},
        }
        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs


class SpeechActivityDetection(LabelingTask):
    """Train speech activity (and overlap) detection

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            frame_info=None, frame_crop=None):
        """
        frame_info : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        frame_crop : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return SpeechActivityDetectionGenerator(
            feature_extraction,
            protocol, subset=subset,
            frame_info=frame_info,
            frame_crop=frame_crop,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel)


class DomainAwareSpeechActivityDetection(SpeechActivityDetection):
    """Domain-aware speech activity detection

    Trains speech activity detection and domain classification jointly.

    Parameters
    ----------
    domain : `str`, optional
        Batch key to use as domain. Defaults to 'domain'.
        Could be 'database' or 'uri' for instance.
    attachment : `int`, optional
        Intermediate level where to attach the domain classifier.
        Defaults to -1. Passed to `return_intermediate` in models supporting it.
    rnn: `dict`, optional 
        Parameters of the RNN used in the domain classifier.
        See `pyannote.audio.models.models.RNN` for details. 
    domain_loss: `str`, optional
        Loss function to use. Defaults to 'NLLLoss'.
    """

    DOMAIN_PT = '{log_dir}/weights/{epoch:04d}.domain.pt'

    def __init__(self, 
                 domain='domain', attachment=-1, 
                 rnn=None, domain_loss="NLLLoss", 
                 **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.attachment = attachment

        if rnn is None:
            rnn = dict()
        self.rnn = rnn

        self.domain_loss = domain_loss
        if self.domain_loss == "NLLLoss":
            # Default value
            self.domain_loss_ = nn.NLLLoss()
            self.activation_ = nn.LogSoftmax(dim=1)
            
        elif self.domain_loss == "MSELoss":
            self.domain_loss_ = nn.MSELoss()
            self.activation_ = nn.Sigmoid()
        
        else:
            msg = (
                f'{domain_loss} has not been implemented yet.'
            )
            raise NotImplementedError(msg)

    def parameters(self, model, specifications, device):
        """Initialize trainable trainer parameters

        Parameters
        ----------
        specifications : `dict`
            Batch specs.

        Returns
        -------
        parameters : iterable
            Trainable trainer parameters
        """
        domain_classifier_rnn = RNN(
            n_features=model.intermediate_dimension(self.attachment), 
            **self.rnn)

        domain_classifier_linear = nn.Linear(
            domain_classifier_rnn.dimension,
            len(specifications[self.domain]['classes']),
            bias=True).to(device)

        self.domain_classifier_ = nn.Sequential(domain_classifier_rnn, 
                                                domain_classifier_linear).to(device)

        return list(self.domain_classifier_.parameters())

    def load_epoch(self, epoch):
        """Load model and classifier from disk

        Parameters
        ----------
        epoch : `int`
            Epoch number.
        """

        super().load_epoch(epoch)

        domain_classifier_state = torch.load(
            self.DOMAIN_PT.format(log_dir=self.log_dir_, epoch=epoch),
            map_location=lambda storage, loc: storage)
        self.domain_classifier_.load_state_dict(domain_classifier_state)

    def save_epoch(self, epoch=None):
        """Save model to disk

        Parameters
        ----------
        epoch : `int`, optional
            Epoch number. Defaults to self.epoch_

        """

        if epoch is None:
            epoch = self.epoch_

        torch.save(self.domain_classifier_.state_dict(),
                   self.DOMAIN_PT.format(log_dir=self.log_dir_,
                                             epoch=epoch))

        super().save_epoch(epoch=epoch)

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)
        fX, intermediate = self.model_(X, return_intermediate=self.attachment)

        # speech activity detection
        fX = fX.view((-1, self.n_classes_))
        target = torch.tensor(
            batch['y'],
            dtype=torch.int64,
            device=self.device_).contiguous().view((-1, ))

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)
        loss = self.loss_func_(fX, target, weight=weight)

        # domain classification
        domain_target = torch.tensor(
            batch[self.domain],
            dtype=torch.int64,
            device=self.device_)

        domain_scores = self.activation_(self.domain_classifier_(intermediate))

        domain_loss = self.domain_loss_(domain_scores, domain_target)

        return {'loss': loss + domain_loss,
                'loss_domain': domain_loss,
                'loss_task': loss}


class DomainAdversarialSpeechActivityDetection(DomainAwareSpeechActivityDetection):
    """Domain Adversarial speech activity detection

    Parameters
    ----------
    domain : `str`, optional
        Batch key to use as domain. Defaults to 'domain'.
        Could be 'database' or 'uri' for instance.
    attachment : `int`, optional
        Intermediate level where to attach the domain classifier.
        Defaults to -1. Passed to `return_intermediate` in models supporting it.
    alpha : `float`, optional
        Coefficient multiplied with the domain loss
    """

    def __init__(self, domain='domain', attachment=-1, alpha=1., **kwargs):
        super().__init__(domain=domain, attachment=attachment, **kwargs)
        self.alpha = alpha
        self.gradient_reversal_ = GradientReversal()

    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """
        # forward pass
        X = torch.tensor(batch['X'],
                         dtype=torch.float32,
                         device=self.device_)

        fX, intermediate = self.model_(X, return_intermediate=self.attachment)

        # speech activity detection
        fX = fX.view((-1, self.n_classes_))

        target = torch.tensor(
            batch['y'],
            dtype=torch.int64,
            device=self.device_).contiguous().view((-1, ))

        weight = self.weight
        if weight is not None:
            weight = weight.to(device=self.device_)

        loss = self.loss_func_(fX, target, weight=weight)

        # domain classification
        domain_target = torch.tensor(
            batch[self.domain],
            dtype=torch.int64,
            device=self.device_)

        domain_scores = self.activation_(self.domain_classifier_(
            self.gradient_reversal_(intermediate)))

        if self.domain_loss == "MSELoss":
            # One hot encode domain_target for Mean Squared Error Loss
            nb_domains = domain_scores.shape[1]
            identity_mat = torch.sparse.torch.eye(nb_domains, device=self.device_)
            domain_target = identity_mat.index_select(dim=0, index=domain_target)

        domain_loss = self.domain_loss_(domain_scores, domain_target)

        return {'loss': loss + self.alpha * domain_loss,
                'loss_domain': domain_loss,
                'loss_task': loss}
