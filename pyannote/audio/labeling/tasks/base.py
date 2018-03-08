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

import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable
from pyannote.audio.checkpoint import Checkpoint
from pyannote.metrics.binary_classification import det_curve


class LabelingTask(object):
    """

    Parameters
    ----------
    duration : float, optional
        Defautls to 3.2 seconds.
    batch_size : int, optional
        Defaults to 32.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, duration=3.2, batch_size=32, parallel=1):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.parallel = parallel

    def get_batch_generator(self, precomputed):
        """Get batch generator"""
        msg = 'LabelingTask subclass must define `get_batch_generator` method.'
        raise NotImplementedError(msg)

    @property
    def n_classes(self):
        """Number of classes"""
        msg = 'LabelingTask subclass must define `n_classes` property.'
        raise NotImplementedError(msg)


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

        if model.n_classes != self.n_classes:
            raise ValueError('n_classes mismatch')
        n_classes = model.n_classes

        checkpoint = Checkpoint(log_dir, restart=restart > 0)

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        batch_generator = self.get_batch_generator(feature_extraction)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)

        batches_per_epoch = batch_generator.batches_per_epoch

        if restart > 0:
            weights_pt = checkpoint.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))

        if gpu:
            model = model.cuda()

        optimizer = Adam(model.parameters())
        if restart > 0:
            optimizer_pt = checkpoint.OPTIMIZER_PT.format(log_dir=log_dir,
                                                    epoch=restart)
            optimizer.load_state_dict(torch.load(optimizer_pt))
            if gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        loss_func = model.get_loss()

        epoch = restart if restart > 0 else -1
        while True:
            epoch += 1
            if epoch > epochs:
                break

            loss_avg = 0.
            log_epoch = (epoch < 10) or (epoch % 5 == 0)

            if log_epoch:
                log_y_pred = []
                log_y_true = []

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = batch['X']
                y = batch['y']
                if not getattr(model, 'batch_first', True):
                    X = np.rollaxis(X, 0, 2)
                    y = y.T
                X = np.array(X, dtype=np.float32)
                X = Variable(torch.from_numpy(X))
                y = Variable(torch.from_numpy(y))

                if gpu:
                    X = X.cuda()
                    y = y.cuda()

                fX = model(X)

                if log_epoch:
                    y_pred = fX.data
                    y_true = y.data
                    if gpu:
                        y_pred = y_pred.cpu()
                        y_true = y_true.cpu()
                    y_pred = y_pred.numpy()
                    y_true = y_true.numpy()
                    log_y_pred.append(y_pred)
                    log_y_true.append(y_true)

                losses = loss_func(fX.view((-1, n_classes)),
                                   y.contiguous().view((-1, )))
                loss = torch.mean(losses)

                # log loss
                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                loss_avg += loss_

                loss.backward()
                optimizer.step()

            loss_avg /= batches_per_epoch
            writer.add_scalar('train/loss', loss_avg,
                              global_step=epoch)

            if log_epoch:
                log_y_pred = np.hstack(log_y_pred)
                log_y_true = np.hstack(log_y_true)
                log_y_pred = log_y_pred.reshape((-1, n_classes))
                log_y_true = log_y_true.reshape((-1, ))
                for k in range(n_classes):
                    _, _, _, eer = det_curve(log_y_true == k, log_y_pred[:, k])
                    writer.add_scalar(f'train/estimate/eer/{k}', eer,
                                      global_step=epoch)

            checkpoint.on_epoch_end(epoch, model, optimizer)
