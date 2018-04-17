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
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.checkpoint import Checkpoint
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop
from pyannote.audio.embedding.utils import to_condensed, pdist, l2_normalize
from scipy.spatial.distance import squareform
from pyannote.metrics.binary_classification import det_curve

from pyannote.audio.util import DavisKingScheduler


class TripletLoss(object):
    """

    delta = d(anchor, positive) - d(anchor, negative)

    * with 'positive' clamping:
        loss = max(0, delta + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * delta)

    where d(., .) varies in range [0, D] (e.g. D=2 for euclidean distance).

    Parameters
    ----------
    duration : float, optional
        Defautls to 3.2 seconds.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin multiplicative factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard', 'negative'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    optimizer : {'sgd', 'rmsprop', 'adam'}
        Defaults to 'rmsprop'.

    """

    def __init__(self, duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None, parallel=1,
                 optimizer='rmsprop'):

        super(TripletLoss, self).__init__()

        self.metric = metric
        self.margin = margin

        self.margin_ = self.margin * self.max_distance

        if clamp not in {'positive', 'sigmoid', 'softmargin'}:
            msg = "'clamp' must be one of {'positive', 'sigmoid', 'softmargin'}."
            raise ValueError(msg)
        self.clamp = clamp

        if sampling not in {'all', 'hard', 'negative'}:
            msg = "'sampling' must be one of {'all', 'hard', 'negative'}."
            raise ValueError(msg)
        self.sampling = sampling

        self.per_fold = per_fold
        self.per_label = per_label
        self.duration = duration
        self.parallel = parallel
        self.optimizer = optimizer

    @property
    def max_distance(self):
        if self.metric == 'cosine':
            return 2.
        elif self.metric == 'angular':
            return np.pi
        elif self.metric == 'euclidean':
            # FIXME. incorrect if embedding are not unit-normalized
            return 2.
        else:
            msg = "'metric' must be one of {'euclidean', 'cosine', 'angular'}."
            raise ValueError(msg)

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.autograd.Variable
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix
        """

        n_sequences, _ = fX.size()
        distances = []

        for i in range(n_sequences - 1):

            if self.metric in ('cosine', 'angular'):
                d = 1. - F.cosine_similarity(
                    fX[i, :].expand(n_sequences - 1 - i, -1),
                    fX[i+1:, :], dim=1, eps=1e-8)

                if self.metric == 'angular':
                    d = torch.acos(torch.clamp(1. - d, -1 + 1e-6, 1 - 1e-6))

            elif self.metric == 'euclidean':
                d = F.pairwise_distance(
                    fX[i, :].expand(n_sequences - 1 - i, -1),
                    fX[i+1:, :], p=2, eps=1e-06).view(-1)

            distances.append(d)

        return torch.cat(distances)

    def batch_hard(self, y, distances):
        """Build triplet with both hardest positive and hardest negative

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        if distances.is_cuda:
            distances = squareform(distances.data.cpu().numpy())
        else:
            distances = squareform(distances.data.numpy())
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            d = distances[anchor]

            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

    def batch_negative(self, y, distances):
        """Build triplet with hardest negative

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        if distances.is_cuda:
            distances = squareform(distances.data.cpu().numpy())
        else:
            distances = squareform(distances.data.numpy())
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        return anchors, positives, negatives

    def batch_all(self, y, distances):
        """Build all possible triplet

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def triplet_loss(self, distances, anchors, positives, negatives,
                     return_delta=False):
        """Compute triplet loss

        Parameters
        ----------
        distances : torch.autograd.Variable
            Condensed matrix of pairwise distances.
        anchors, positives, negatives : list of int
            Triplets indices.
        return_delta : bool, optional
            Return delta before clamping.

        Returns
        -------
        loss : torch.autograd.Variable
            Triplet loss.
        """

        # estimate total number of embeddings from pdist shape
        n = int(.5 * (1 + np.sqrt(1 + 8 * len(distances))))
        n = [n] * len(anchors)

        # convert indices from squared matrix
        # to condensed matrix referential
        pos = list(map(to_condensed, n, anchors, positives))
        neg = list(map(to_condensed, n, anchors, negatives))

        # compute raw triplet loss (no margin, no clamping)
        # the lower, the better
        delta = distances[pos] - distances[neg]

        # clamp triplet loss
        if self.clamp == 'positive':
            loss = torch.clamp(delta + self.margin_, min=0)

        elif self.clamp == 'softmargin':
            loss = torch.log1p(torch.exp(delta))

        elif self.clamp == 'sigmoid':
            # TODO. tune this "10" hyperparameter
            loss = F.sigmoid(10 * (delta + self.margin_))

        # return triplet losses
        if return_delta:
            return loss, delta.view((-1, 1)), pos, neg
        else:
            return loss

    def get_batch_generator(self, feature_extraction):
        return SpeechSegmentGenerator(
            feature_extraction,
            per_label=self.per_label, per_fold=self.per_fold,
            duration=self.duration, parallel=self.parallel)

    def aggregate(self, batch):
        return batch

    def backtrack(self, epoch):
        """Backtrack to `epoch` state

        This assumes that the following attributes have been set already:

        * checkpoint_
        * model_
        * gpu_
        * batches_per_epoch_

        This will set/update the following hidden attributes:

        * model_
        * optimizer_
        * scheduler_

        """

        if epoch > 0:
            weights_pt = self.checkpoint_.weights_pt(epoch)
            self.model_.load_state_dict(torch.load(weights_pt))

        if self.gpu_:
            self.model_ = self.model_.cuda()

        if self.optimizer == 'sgd':
            self.optimizer_ = SGD(self.model_.parameters(), lr=1e-2,
                                  momentum=0.9, nesterov=True)

        elif self.optimizer == 'adam':
            self.optimizer_ = Adam(self.model_.parameters())

        elif self.optimizer == 'rmsprop':
            self.optimizer_ = RMSprop(self.model_.parameters())

        self.model_.internal = False

        if epoch > 0:
            optimizer_pt = self.checkpoint_.optimizer_pt(epoch)
            self.optimizer_.load_state_dict(torch.load(optimizer_pt))
            if self.gpu_:
                for state in self.optimizer_.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        self.scheduler_ = DavisKingScheduler(
            self.optimizer_, self.batches_per_epoch_, factor=0.5,
            patience=10, active=self.optimizer == 'sgd')

    def to_numpy(self, variable):
        if self.gpu_:
            return variable.data.cpu().numpy()
        return variable.data.numpy()

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=0, gpu=False):
        """Train model

        Parameters
        ----------
        model : nn.Module
            Embedding model
        feature_extraction :
            Feature extraction.
        protocol : pyannote.database.Protocol
        log_dir : str
            Directory where models and other log files are stored.
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'train'.
        epochs : int, optional
            Train model for that many epochs.
        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        gpu : bool, optional
        """

        # initialize tensorboard support
        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        self.checkpoint_ = Checkpoint(
            log_dir=log_dir, restart=restart > 0)
        self.model_ = model
        self.gpu_ = gpu

        batch_generator = self.get_batch_generator(feature_extraction)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)
        self.batches_per_epoch_ = batch_generator.batches_per_epoch

        # initialize model, optimizer, and scheduler
        self.backtrack(restart)

        epoch = restart if restart > 0 else -1
        backtrack, iteration = False, 0

        while True:
            # keep track of actual number of iterations
            iteration += 1

            # keep track of current epoch
            # due to backtracking, this may lag a bit behind `iteration`
            epoch += 1
            if epoch > epochs:
                break

            # log backtracking
            writer.add_scalar('train/scheduler/backtrack', epoch,
                              global_step=iteration)

            tloss_avg = 0.

            # detailed logging to Tensorboard
            # for first 10 epochs then every other 5 epochs
            detailed_log = (iteration < 10) or (iteration % 5 == 0)

            if detailed_log:
                log_positive = []
                log_negative = []
                log_delta = []
                log_norm = []

            desc = 'Epoch #{0}'.format(iteration)
            for i in tqdm(range(self.batches_per_epoch_), desc=desc):

                # zero gradients
                self.model_.zero_grad()

                # process next batch
                batch = next(batches)

                # update batch['X'] to be usable by torch
                X = batch['X']
                if not getattr(self.model_, 'batch_first', True):
                    X = np.rollaxis(X, 0, 2)
                X = np.array(X, dtype=np.float32)
                X = Variable(torch.from_numpy(X))
                if gpu:
                    X = X.cuda()
                batch['X'] = X

                # forward pass
                fX = self.model_(batch['X'])

                # log embedding norms
                if detailed_log:
                    norm_npy = np.linalg.norm(self.to_numpy(fX), axis=1)
                    log_norm.append(norm_npy)

                batch['fX'] = fX
                batch = self.aggregate(batch)

                fX = batch['fX']
                y = batch['y']

                # pre-compute pairwise distances
                distances = self.pdist(fX)

                # sample triplets
                triplets = getattr(self, 'batch_{0}'.format(self.sampling))
                anchors, positives, negatives = triplets(y, distances)

                # compute loss for each triplet
                losses, deltas, _, _ = self.triplet_loss(
                    distances, anchors, positives, negatives,
                    return_delta=True)
                # average over all triplets
                loss = torch.mean(losses)

                if detailed_log:
                    pdist_npy = self.to_numpy(distances)
                    delta_npy = self.to_numpy(deltas)
                    same_speaker = pdist(y.reshape((-1, 1)), metric='equal')
                    log_positive.append(pdist_npy[np.where(same_speaker)])
                    log_negative.append(pdist_npy[np.where(~same_speaker)])
                    log_delta.append(delta_npy)

                # accumulate loss over the whole epoch
                loss_ = float(self.to_numpy(loss))
                tloss_avg += loss_

                # back-propagation
                loss.backward()
                self.optimizer_.step()

                # send loss of current batch to scheduler
                # and receive information about loss trend
                scheduler_state = self.scheduler_.step(loss_)

                # log loss trend statistics to tensorboard
                for name, value in scheduler_state.items():
                    writer.add_scalar(
                        f'train/scheduler/{name}', value,
                        global_step=iteration * self.batches_per_epoch_ + i)
                writer.add_scalar(
                    f'train/scheduler/loss', loss_,
                    global_step=iteration * self.batches_per_epoch_ + i)

                # remember to backtrack after the epoch has completed
                # in case it looks like loss is increasing
                if scheduler_state['increasing_probability'] > 0.99:
                    backtrack = True

            # log loss to tensorboard
            writer.add_scalar('train/triplet/loss',
                              tloss_avg / self.batches_per_epoch_,
                              global_step=iteration)

            # log current learning rate to tensorboard
            writer.add_scalar('train/scheduler/lr',
                              self.scheduler_.lr[0],
                              global_step=iteration)

            # # log loss trend statistics to tensorboard
            # for name, value in scheduler_state.items():
            #     writer.add_scalar(f'train/scheduler/{name}',
            #                       value, global_step=iteration)

            if detailed_log:

                # log intra class vs. inter class distance distributions
                log_positive = np.hstack(log_positive)
                log_negative = np.hstack(log_negative)
                bins = np.linspace(0, self.max_distance, 50)
                try:
                    writer.add_histogram(
                        'train/distance/intra_class', log_positive,
                        global_step=iteration, bins=bins)
                    writer.add_histogram(
                        'train/distance/inter_class', log_negative,
                        global_step=iteration, bins=bins)
                except ValueError as e:
                    pass

                # log same/different experiment on training samples
                _, _, _, eer = det_curve(
                    np.hstack([np.ones(len(log_positive)),
                               np.zeros(len(log_negative))]),
                    np.hstack([log_positive, log_negative]),
                    distances=True)
                writer.add_scalar('train/estimate/eer', eer,
                                  global_step=iteration)

                # log raw triplet loss (before max(0, .))
                log_delta = np.vstack(log_delta)
                bins = np.linspace(-self.max_distance, self.max_distance, 50)
                try:
                    writer.add_histogram(
                        'train/triplet/delta', log_delta,
                        global_step=iteration, bins=bins)
                except ValueError as e:
                    pass

                # log distribution of embedding norms
                log_norm = np.hstack(log_norm)
                try:
                    writer.add_histogram(
                        'train/embedding/norm', log_norm,
                        global_step=iteration, bins='doane')
                except ValueError as e:
                    pass

            # save model weights (and optimizer state) to disk
            self.checkpoint_.on_epoch_end(iteration, self.model_,
                                          self.optimizer_)

            # backtrack in case loss has increased
            if backtrack:
                backtrack = False
                epoch = max(0, epoch - 3)
                self.backtrack(epoch)
