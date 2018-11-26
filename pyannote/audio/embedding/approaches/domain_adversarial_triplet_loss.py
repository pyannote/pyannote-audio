#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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
from tqdm import tqdm
import torch
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.nn as nn
from pyannote.audio.embedding.generators import SpeechSegmentGenerator
from pyannote.audio.checkpoint import Checkpoint
from torch.optim import Adam
from .triplet_loss import TripletLoss
from scipy.spatial.distance import pdist, squareform


class GradReverse(Function):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.alpha)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class DomainClassifier(nn.Module):

    def __init__(self, n_dimensions, n_domains, alpha=1.):
        super(DomainClassifier, self).__init__()
        n_hidden = 100
        self.fc1 = nn.Linear(n_dimensions, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_domains)
        # self.drop = nn.Dropout2d(0.25)
        self.alpha = alpha

    def forward(self, x):
        x = grad_reverse(x, self.alpha)
        # x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)

class DomainAdversarialTripletLoss(TripletLoss):
    """

    delta = d(anchor, positive) - d(anchor, negative)

    * with 'positive' clamping:
        loss = max(0, delta + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * delta)

    where d(., .) varies in range [0, D] (e.g. D=2 for euclidean distance).

    Parameters
    ----------
    domain : {'database'}, optional
        Domain. Typically 'database'.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    """

    def __init__(self, domain='database', duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None):

        super(DomainAdversarialTripletLoss, self).__init__(
            duration=duration, metric=metric, margin=margin, clamp=clamp,
            sampling=sampling, per_label=per_label, per_fold=per_fold)
        self.domain = domain

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=None, gpu=False):

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        checkpoint = Checkpoint(
            log_dir=log_dir, restart=(False if restart is None else True))
        try:
            batch_generator = SpeechSegmentGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)
        except OSError as e:
            del batch_generator.data_
            batch_generator = SpeechSegmentGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration, fast=False)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)

        # one minute per speaker
        duration_per_epoch = 60. * batch_generator.n_labels
        duration_per_batch = self.duration * batch_generator.n_sequences_per_batch
        batches_per_epoch = int(np.ceil(duration_per_epoch / duration_per_batch))

        if restart is not None:
            weights_pt = checkpoint.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))

        if gpu:
            model = model.cuda()

        model.internal = False

        n_domains = len(batch_generator.domains_[self.domain])
        if n_domains < 2:
            raise ValueError('There must be more than one domain.')

        domain_clf = DomainClassifier(model.output_dim, n_domains, alpha=1.)
        if gpu:
            domain_clf = domain_clf.cuda()

        domain_loss = nn.CrossEntropyLoss()

        optimizer = Adam(list(model.parameters()) + list(domain_clf.parameters()))
        if restart is not None:
            optimizer_pt = checkpoint.OPTIMIZER_PT.format(
                log_dir=log_dir, epoch=restart)
            optimizer.load_state_dict(torch.load(optimizer_pt))
            if gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        restart = 0 if restart is None else restart + 1
        for epoch in range(restart, restart + epochs):

            tloss_avg = 0.
            dloss_avg = 0.
            loss_avg = 0.
            dacc_avg = 0.
            positive, negative = [], []
            if not model.normalize:
                norms = []

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = batch['X']
                if not getattr(model, 'batch_first', True):
                    X = np.rollaxis(X, 0, 2)
                X = np.array(X, dtype=np.float32)
                X = Variable(torch.from_numpy(X))

                y = batch['y']
                y_domain = batch['y_{domain}'.format(domain=self.domain)]

                if gpu:
                    X = X.cuda()
                fX = model(X)

                if not model.normalize:
                    if gpu:
                        fX_ = fX.data.cpu().numpy()
                    else:
                        fX_ = fX.data.numpy()
                    norms.append(np.linalg.norm(fX_, axis=0))

                triplet_losses = []
                for d, domain in enumerate(np.unique(y_domain)):

                    this_domain = np.where(y_domain == domain)[0]

                    domain_y = y[this_domain]

                    # if there is less than 2 speakers in this domain, skip it
                    if len(np.unique(domain_y)) < 2:
                        continue

                    # pre-compute within-domain pairwise distances
                    domain_fX = fX[this_domain, :]
                    domain_pdist = self.pdist(domain_fX)

                    if gpu:
                        domain_pdist_ = domain_pdist.data.cpu().numpy()
                    else:
                        domain_pdist_ = domain_pdist.data.numpy()
                    is_positive = pdist(domain_y.reshape((-1, 1)), metric='chebyshev') < 1
                    positive.append(domain_pdist_[np.where(is_positive)])
                    negative.append(domain_pdist_[np.where(~is_positive)])

                    # sample triplets
                    if self.sampling == 'all':
                        anchors, positives, negatives = self.batch_all(
                            domain_y, domain_pdist)

                    elif self.sampling == 'hard':
                        anchors, positives, negatives = self.batch_hard(
                            domain_y, domain_pdist)

                    # compute triplet loss
                    triplet_losses.append(
                        self.triplet_loss(domain_pdist, anchors,
                                          positives, negatives))

                tloss = 0.
                for tl in triplet_losses:
                    tloss += torch.mean(tl)
                tloss /= len(triplet_losses)

                if gpu:
                    tloss_ = float(tloss.data.cpu().numpy())
                else:
                    tloss_ = float(tloss.data.numpy())
                tloss_avg += tloss_

                # domain-adversarial
                y_domain = Variable(torch.from_numpy(np.array(y_domain)))
                if gpu:
                    y_domain = y_domain.cuda()

                domain_score = domain_clf(fX)
                dloss = domain_loss(domain_score, y_domain)

                if gpu:
                    dloss_ = float(dloss.data.cpu().numpy())
                else:
                    dloss_ = float(dloss.data.numpy())
                dloss_avg += dloss_

                # log domain classification accuracy
                if gpu:
                    domain_score_ = domain_score.data.cpu().numpy()
                    y_domain_ = y_domain.data.cpu().numpy()
                else:
                    domain_score_ = domain_score.data.numpy()
                    y_domain_ = y_domain.data.numpy()
                dacc_ = np.mean(np.argmax(domain_score_, axis=1) == y_domain_)
                dacc_avg += dacc_

                loss = tloss + dloss

                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                loss_avg += loss_
                loss_avg += loss_

                loss.backward()
                optimizer.step()

            # if gpu:
            #     embeddings = fX.data.cpu()
            # else:
            #     embeddings = fX.data
            # metadata = list(batch['extra'][self.domain])
            #
            # writer.add_embedding(embeddings, metadata=metadata,
            #                      global_step=epoch)

            tloss_avg /= batches_per_epoch
            writer.add_scalar('tloss', tloss_avg, global_step=epoch)
            dloss_avg /= batches_per_epoch
            writer.add_scalar('dloss', dloss_avg, global_step=epoch)
            loss_avg /= batches_per_epoch
            writer.add_scalar('loss', loss_avg, global_step=epoch)
            dacc_avg /= batches_per_epoch
            writer.add_scalar('dacc', dacc_avg, global_step=epoch)

            positive = np.hstack(positive)
            negative = np.hstack(negative)
            writer.add_histogram(
                'embedding/pairwise_distance/positive', positive,
                global_step=epoch, bins='auto')
            writer.add_histogram(
                'embedding/pairwise_distance/negative', negative,
                global_step=epoch, bins='auto')

            if not model.normalize:
                norms = np.hstack(norms)
                writer.add_histogram(
                    'embedding/norm', norms,
                    global_step=epoch, bins='auto')

            checkpoint.on_epoch_end(epoch, model, optimizer)
