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
import torch.nn as nn
from torch.autograd import Variable
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.checkpoint import Checkpoint
from torch.optim import Adam


class Classifier(nn.Module):
    """MLP classifier

    Parameters
    ----------
    n_dimensions : int
        Embedding dimension
    n_classes : int
        Number of classes.
    linear : list, optional
        By default, classifier is just a (n_dimension > n_classes) linear layer
        followed by a softmax. Use this option to provide hidden dimensions of
        (optional) additional linear layer (e.g. [100, 1000] to 3 layers:
        n_dimension > 100 > 1000 > n_classes).
    """

    def __init__(self, n_dimensions, n_classes, linear=[]):
        super(Classifier, self).__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.linear = linear

        # create list of linear layers
        self.linear_layers_ = []
        input_dim = self.n_dimensions
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        final_layer = nn.Linear(input_dim, self.n_classes, bias=True)
        self.final_layer_ = final_layer

        self.tanh_ = nn.Tanh()
        self.logsoftmax_ = nn.LogSoftmax(dim=1)

    def forward(self, embedding):

        output = embedding

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = self.tanh_(output)

        # apply final linear layer
        output = self.final_layer_(output)
        output = self.tanh_(output)
        output = self.logsoftmax_(output)

        return output


class Softmax(object):
    """Train embeddings in a supervised (classification) manner

    Parameters
    ----------
    duration : float, optional
        Defautls to 3.2 seconds.
    per_fold : int, optional
        Number of speakers per batch. Defaults to the whole speaker set.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 1.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    linear : list, optional
        By default, classifier is just a (n_dimension > n_classes) linear layer
        followed by a softmax. Use this option to provide hidden dimensions of
        (optional) additional linear layer (e.g. [100, 1000] to 3 layers:
        n_dimension > 100 > 1000 > n_classes).
    """

    CLASSES_TXT = '{log_dir}/classes.txt'
    CLASSIFIER_PT = '{log_dir}/weights/{epoch:04d}.classifier.pt'

    def __init__(self, duration=3.2, per_label=1,
                 per_fold=None, parallel=1, linear=[]):
        super(Softmax, self).__init__()
        self.per_fold = per_fold
        self.per_label = per_label
        self.duration = duration
        self.parallel = parallel
        self.linear = linear
        self.loss_ = nn.NLLLoss()

    def get_batch_generator(self, feature_extraction):
        return SpeechSegmentGenerator(
            feature_extraction,
            per_label=self.per_label, per_fold=self.per_fold,
            duration=self.duration, parallel=self.parallel)

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

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        checkpoint = Checkpoint(log_dir=log_dir,
                                      restart=restart > 0)

        batch_generator = self.get_batch_generator(feature_extraction)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)

        batches_per_epoch = batch_generator.batches_per_epoch

        # save list of classes (one speaker per line)
        labels = batch_generator.labels
        classes_txt = self.CLASSES_TXT.format(log_dir=log_dir)
        with open(classes_txt, mode='w') as fp:
            for label in labels:
                fp.write(f'{label}\n')

        # initialize classifier
        n_classes = batch_generator.n_classes
        classifier = Classifier(model.output_dim, n_classes,
                                linear=self.linear)

        # load precomputed weights in case of restart
        if restart > 0:
            weights_pt = checkpoint.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))
            classifier_pt = self.CLASSIFIER_PT.format(
                log_dir=log_dir, epoch=restart)

        # send models to GPU
        if gpu:
            model = model.cuda()
            classifier = classifier.cuda(device=None)

        model.internal = False

        optimizer = Adam(list(model.parameters()) + \
                         list(classifier.parameters()))
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

            loss_avg = 0.

            log_epoch = (epoch < 10) or (epoch % 5 == 0)

            if log_epoch:
                pass

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = batch['X']
                y = batch['y']
                if not getattr(model, 'batch_first', True):
                    X = np.rollaxis(X, 0, 2)
                X = np.array(X, dtype=np.float32)
                X = Variable(torch.from_numpy(X))
                y = Variable(torch.from_numpy(y))

                if gpu:
                    X = X.cuda()
                    y = y.cuda()

                fX = model(X)
                y_pred = classifier(fX)

                loss = self.loss_(y_pred, y)

                if log_epoch:
                    pass

                # log loss
                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                loss_avg += loss_

                loss.backward()
                optimizer.step()

            loss_avg /= batches_per_epoch
            writer.add_scalar('train/softmax/loss', loss_avg,
                              global_step=epoch)

            if log_epoch:
                pass

            checkpoint.on_epoch_end(epoch, model, optimizer,
                                    extra={self.CLASSIFIER_PT: classifier})
