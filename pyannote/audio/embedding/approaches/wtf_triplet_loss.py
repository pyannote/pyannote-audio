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
import torch.nn as nn
import torch.nn.functional as F
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.checkpoint import Checkpoint
from torch.optim import Adam
from scipy.spatial.distance import pdist
from .triplet_loss import TripletLoss
from pyannote.metrics.binary_classification import det_curve


class WTFTripletLoss(TripletLoss):
    """

    Parameters
    ----------
    variant : int, optional
        Loss variants. Defaults to 1.
    duration : float, optional
        Defautls to 3.2 seconds.
    margin: float, optional
        Margin factor. Defaults to 0.2.
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
    """

    CONFIDENCE_PT = '{log_dir}/weights/{epoch:04d}.confidence.pt'

    def __init__(self, variant=1, duration=3.2, sampling='all',
                 per_label=3, per_fold=None, parallel=1):

        super(WTFTripletLoss, self).__init__(
            duration=duration, metric='angular', clamp='sigmoid',
            sampling=sampling, per_label=per_label, per_fold=per_fold,
            parallel=parallel)

        self.variant = variant


    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=0, gpu=False):

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        checkpoint = Checkpoint(log_dir=log_dir,
                                      restart=restart > 0)

        batch_generator = SpeechSegmentGenerator(
            feature_extraction,
            per_label=self.per_label, per_fold=self.per_fold,
            duration=self.duration, parallel=self.parallel)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)

        batches_per_epoch = batch_generator.batches_per_epoch

        if restart > 0:
            weights_pt = checkpoint.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))

        if gpu:
            model = model.cuda()

        model.internal = False

        parameters = list(model.parameters())

        if self.variant in [2, 3, 4, 5, 6, 7, 8]:

            # norm batch-normalization
            self.norm_bn = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=True)
            if gpu:
                self.norm_bn = self.norm_bn.cuda()
            parameters += list(self.norm_bn.parameters())

        if self.variant in [9]:
            # norm batch-normalization
            self.norm_bn = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=False)
            if gpu:
                self.norm_bn = self.norm_bn.cuda()
            parameters += list(self.norm_bn.parameters())

        if self.variant in [5, 6, 7]:
            self.positive_bn = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=False)
            self.negative_bn = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=False)
            if gpu:
                self.positive_bn = self.positive_bn.cuda()
                self.negative_bn = self.negative_bn.cuda()
            parameters += list(self.positive_bn.parameters())
            parameters += list(self.negative_bn.parameters())

        if self.variant in [8, 9]:

            self.delta_bn = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=False)
            if gpu:
                self.delta_bn = self.delta_bn.cuda()
            parameters += list(self.delta_bn.parameters())

        optimizer = Adam(parameters)
        if restart > 0:
            optimizer_pt = checkpoint.OPTIMIZER_PT.format(
                log_dir=log_dir, epoch=restart)
            optimizer.load_state_dict(torch.load(optimizer_pt))
            if gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        epoch = restart if restart > 0 else -1
        while True:
            epoch += 1
            if epoch > epochs:
                break

            loss_avg, tloss_avg, closs_avg = 0., 0., 0.

            if epoch % 5 == 0:
                log_positive = []
                log_negative = []
                log_delta = []
                log_norm = []

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = batch['X']
                if not getattr(model, 'batch_first', True):
                    X = np.rollaxis(X, 0, 2)
                X = np.array(X, dtype=np.float32)
                X = Variable(torch.from_numpy(X))

                if gpu:
                    X = X.cuda()

                fX = model(X)

                # pre-compute pairwise distances
                distances = self.pdist(fX)

                # sample triplets
                triplets = getattr(self, 'batch_{0}'.format(self.sampling))
                anchors, positives, negatives = triplets(batch['y'], distances)

                # compute triplet loss
                tlosses, deltas, pos_index, neg_index  = self.triplet_loss(
                    distances, anchors, positives, negatives,
                    return_delta=True)

                tloss = torch.mean(tlosses)

                if self.variant == 1:

                    closses = F.sigmoid(
                        F.softsign(deltas) * torch.norm(fX[anchors], 2, 1, keepdim=True))

                    # if d(a, p) < d(a, n) (i.e. good case)
                    #   --> sign(delta) < 0
                    #   --> loss decreases when norm increases.
                    #       i.e. encourages longer anchor

                    # if d(a, p) > d(a, n) (i.e. bad case)
                    #   --> sign(delta) > 0
                    #   --> loss increases when norm increases
                    #       i.e. encourages shorter anchor

                elif self.variant == 2:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    norms_ = F.sigmoid(self.norm_bn(norms_))

                    confidence = (norms_[anchors] + norms_[positives] + norms_[negatives]) / 3
                    # if |x| is average
                    #    --> normalized |x| = 0
                    #    --> confidence = 0.5

                    # if |x| is bigger than average
                    #    --> normalized |x| >> 0
                    #    --> confidence = 1

                    # if |x| is smaller than average
                    #    --> normalized |x| << 0
                    #    --> confidence = 0

                    correctness = F.sigmoid(-deltas / np.pi * 6)
                    # if d(a, p) = d(a, n) (i.e. uncertain case)
                    #    --> correctness = 0.5

                    # if d(a, p) - d(a, n) = -𝛑 (i.e. best possible case)
                    #    --> correctness = 1

                    # if d(a, p) - d(a, n) = +𝛑 (i.e. worst possible case)
                    #    --> correctness = 0

                    closses = torch.abs(confidence - correctness)
                    # small if (and only if) confidence & correctness agree

                elif self.variant == 3:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    norms_ = F.sigmoid(self.norm_bn(norms_))
                    confidence = (norms_[anchors] * norms_[positives] * norms_[negatives]) / 3

                    correctness = F.sigmoid(-(deltas + np.pi / 4) / np.pi * 6)
                    # correctness = 0.5 at delta == -pi/4
                    # correctness = 1 for delta == -pi
                    # correctness = 0 for delta < 0

                    closses = torch.abs(confidence - correctness)

                elif self.variant == 4:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    norms_ = F.sigmoid(self.norm_bn(norms_))
                    confidence = (norms_[anchors] * norms_[positives] * norms_[negatives]) ** 1/3

                    correctness = F.sigmoid(-(deltas + np.pi / 4) / np.pi * 6)
                    # correctness = 0.5 at delta == -pi/4
                    # correctness = 1 for delta == -pi
                    # correctness = 0 for delta < 0

                    # delta = pos - neg ... should be < 0

                    closses = torch.abs(confidence - correctness)

                elif self.variant == 5:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    confidence = F.sigmoid(self.norm_bn(norms_))

                    confidence_pos = .5 * (confidence[anchors] + confidence[positives])
                    # low positive distance == high correctness
                    correctness_pos = F.sigmoid(
                        -self.positive_bn(distances[pos_index].view(-1, 1)))

                    confidence_neg = .5 * (confidence[anchors] + confidence[negatives])
                    # high negative distance == high correctness
                    correctness_neg = F.sigmoid(
                        self.negative_bn(distances[neg_index].view(-1, 1)))

                    closses = .5 * (torch.abs(confidence_pos - correctness_pos) \
                                  + torch.abs(confidence_neg - correctness_neg))

                elif self.variant == 6:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    confidence = F.sigmoid(self.norm_bn(norms_))

                    confidence_pos = .5 * (confidence[anchors] + confidence[positives])
                    # low positive distance == high correctness
                    correctness_pos = F.sigmoid(
                        -self.positive_bn(distances[pos_index].view(-1, 1)))

                    closses = torch.abs(confidence_pos - correctness_pos)

                elif self.variant == 7:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    confidence = F.sigmoid(self.norm_bn(norms_))

                    confidence_neg = .5 * (confidence[anchors] + confidence[negatives])
                    # high negative distance == high correctness
                    correctness_neg = F.sigmoid(
                        self.negative_bn(distances[neg_index].view(-1, 1)))

                    closses = torch.abs(confidence_neg - correctness_neg)

                elif self.variant in [8, 9]:

                    norms_ = torch.norm(fX, 2, 1, keepdim=True)
                    norms_ = F.sigmoid(self.norm_bn(norms_))
                    confidence = (norms_[anchors] * norms_[positives] * norms_[negatives]) / 3

                    correctness = F.sigmoid(-self.delta_bn(deltas))
                    closses = torch.abs(confidence - correctness)

                closs = torch.mean(closses)

                if epoch % 5 == 0:

                    if gpu:
                        fX_npy = fX.data.cpu().numpy()
                        pdist_npy = distances.data.cpu().numpy()
                        delta_npy = deltas.data.cpu().numpy()
                    else:
                        fX_npy = fX.data.numpy()
                        pdist_npy = distances.data.numpy()
                        delta_npy = deltas.data.numpy()

                    log_norm.append(np.linalg.norm(fX_npy, axis=1))

                    same_speaker = pdist(batch['y'].reshape((-1, 1)), metric='chebyshev') < 1
                    log_positive.append(pdist_npy[np.where(same_speaker)])
                    log_negative.append(pdist_npy[np.where(~same_speaker)])

                    log_delta.append(delta_npy)

                # log loss
                if gpu:
                    tloss_ = float(tloss.data.cpu().numpy())
                    closs_ = float(closs.data.cpu().numpy())
                else:
                    tloss_ = float(tloss.data.numpy())
                    closs_ = float(closs.data.numpy())
                tloss_avg += tloss_
                closs_avg += closs_
                loss_avg += tloss_ + closs_

                loss = tloss + closs
                loss.backward()
                optimizer.step()

            tloss_avg /= batches_per_epoch
            writer.add_scalar('tloss', tloss_avg, global_step=epoch)

            closs_avg /= batches_per_epoch
            writer.add_scalar('closs', closs_avg, global_step=epoch)

            loss_avg /= batches_per_epoch
            writer.add_scalar('loss', loss_avg, global_step=epoch)

            if epoch % 5 == 0:

                log_positive = np.hstack(log_positive)
                writer.add_histogram(
                    'embedding/pairwise_distance/positive', log_positive,
                    global_step=epoch, bins=np.linspace(0, np.pi, 50))
                log_negative = np.hstack(log_negative)

                writer.add_histogram(
                    'embedding/pairwise_distance/negative', log_negative,
                    global_step=epoch, bins=np.linspace(0, np.pi, 50))

                _, _, _, eer = det_curve(
                    np.hstack([np.ones(len(log_positive)), np.zeros(len(log_negative))]),
                    np.hstack([log_positive, log_negative]), distances=True)
                writer.add_scalar('eer', eer, global_step=epoch)

                log_norm = np.hstack(log_norm)
                writer.add_histogram(
                    'norm', log_norm,
                    global_step=epoch, bins='doane')

                log_delta = np.vstack(log_delta)
                writer.add_histogram(
                    'delta', log_delta,
                    global_step=epoch, bins='doane')

            checkpoint.on_epoch_end(epoch, model, optimizer)

            if hasattr(self, 'norm_bn'):
                confidence_pt = self.CONFIDENCE_PT.format(
                    log_dir=log_dir, epoch=epoch)
                torch.save(self.norm_bn.state_dict(), confidence_pt)

    @staticmethod
    def confidence(params, norms):
        """Compute estimated confidence from norm

        Usage
        -----
        >>> params = torch.load('confidence.pt')
        >>> confidence = WTFTripletLoss.confidence(params, norm)
        """

        mu = float(params['running_mean'])
        var = float(params['running_var'])
        eps = 1e-5
        gamma = float(params.get('weight', 1.))
        beta = float(params.get('bias', 0.))

        before_sigmoid = (norms - mu) / np.sqrt(var + eps) * gamma + beta

        return 1 / (1 + np.exp(-before_sigmoid))
