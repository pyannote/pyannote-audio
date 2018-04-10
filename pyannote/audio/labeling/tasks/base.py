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
import random
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable
from pyannote.audio.checkpoint import Checkpoint
from pyannote.metrics.binary_classification import det_curve
from pyannote.database import get_unique_identifier
from pyannote.database import get_label_identifier
from pyannote.database import get_annotated
from pyannote.audio.util import to_numpy
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import SlidingWindowFeature

from pyannote.generators.batch import batchify
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.fragment import SlidingSegments


class LabelingTaskGenerator(object):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed features
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform and unbalanced label distribution).
    """

    def __init__(self, precomputed, duration=3.2, batch_size=32,
                 per_epoch=3600, parallel=1, exhaustive=False):

        super(LabelingTaskGenerator, self).__init__()

        self.precomputed = precomputed
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.exhaustive = exhaustive

    def initialize(self, protocol, subset='train'):
        """Gather the following information about the training subset:

        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        databases_ : list
            Sorted list of (unique) databases in protocol.

        labels_ : list
            Sorted list of (unique) lables in protocol.
        """

        self.data_ = {}
        labels, databases = set(), set()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of database
            database = current_file['database']
            databases.add(database)

            # keep track of unique labels
            for label in current_file['annotation'].labels():
                label = get_label_identifier(label, current_file)
                labels.add(label)

            annotated = get_annotated(current_file)

            if not self.precomputed.use_memmap:
                msg = ('Loading all precomputed features in memory. '
                       'Set "use_memmap" to True if you run out of memory.')
                warnings.warn(msg)

            segments = [s for s in annotated if s.duration > self.duration]

            # corner case where no segment is long enough
            # and we removed them all...
            if not segments:
                continue

            # total duration of label in current_file (after removal of
            # short segments).
            duration = sum(s.duration for s in segments)

            # store all these in data_ dictionary
            datum = {'segments': segments,
                     'duration': duration,
                     'current_file': current_file}
            uri = get_unique_identifier(current_file)
            self.data_[uri] = datum

        self.databases_ = sorted(databases)
        self.labels_ = sorted(labels)

        sliding_window = self.precomputed.sliding_window()
        for current_file in getattr(protocol, subset)():
            y, _ = to_numpy(current_file, self.precomputed,
                            labels=self.labels_)
            uri = get_unique_identifier(current_file)
            self.data_[uri]['y'] = SlidingWindowFeature(
                self.postprocess_y(y), sliding_window)

    def postprocess_y(self, Y):
        """This function does nothing but return its input.
        It should be overriden by subclasses."""
        return Y

    def samples(self):
        if self.exhaustive:
            return self.random_samples()
        else:
            return self.sliding_samples()

    def random_samples(self):
        """Random samples

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        while True:

            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]

            datum = self.data_[uri]
            current_file = datum['current_file']

            # choose one segment at random with probability
            # proportional to its duration
            segment = next(random_segment(datum['segments'], weighted=True))

            # choose fixed-duration subsegment at random
            sequence = next(random_subsegment(segment, self.duration))

            X = self.precomputed.crop(current_file,
                                      sequence, mode='center',
                                      fixed=self.duration)

            y = datum['y'].crop(sequence, mode='center', fixed=self.duration)

            yield {'X': X, 'y': np.squeeze(y)}

    def sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        sliding_segments = SlidingSegments(duration=self.duration,
                                           step=self.duration,
                                           source='annotated')

        while True:

            random.shuffle(uris)

            # loop on all files
            for uri in uris:
                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum['current_file'])

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                annotated = Timeline(
                    [Segment(s.start + random.random() * self.duration,
                             s.end) for s in get_annotated(current_file)])
                current_file['annotated'] = annotated

                for sequence in sliding_segments.from_file(current_file):

                    X = self.precomputed.crop(current_file,
                                              sequence, mode='center',
                                              fixed=self.duration)

                    y = datum['y'].crop(sequence, mode='center',
                                        fixed=self.duration)

                    yield {'X': X, 'y': np.squeeze(y)}

    @property
    def signature(self):
        """Generator signature"""
        return {'X': {'type': 'ndarray'},
                'y': {'type': 'ndarray'}}

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(self.per_epoch / duration_per_batch))

    @property
    def labels(self):
        return list(self.labels_)

    def __call__(self, protocol, subset='train'):
        """(Parallelized) batch generator"""

        # pre-load useful information about protocol once and for all
        self.initialize(protocol, subset=subset)

        # number of batches needed to complete an epoch
        batches_per_epoch = self.batches_per_epoch

        generators = []

        if self.parallel:
            for _ in range(self.parallel):

                # initialize one sample generator
                samples = self.samples()

                # batchify it and make sure at least
                # `batches_per_epoch` batches are prefetched.
                batches = batchify(samples, self.signature,
                                   batch_size=self.batch_size,
                                   prefetch=batches_per_epoch)

                # add batch generator to the list of (background) generators
                generators.append(batches)
        else:

            # initialize one sample generator
            samples = self.samples()

            # batchify it without prefetching
            batches = batchify(samples, self.signature,
                               batch_size=self.batch_size, prefetch=0)

            # add it to the list of generators
            # NOTE: this list will only contain one generator
            generators.append(batches)

        # loop on (background) generators indefinitely
        while True:
            for batches in generators:
                # yield `batches_per_epoch` batches from current generator
                # so that each epoch is covered by exactly one generator
                for _ in range(batches_per_epoch):
                    yield next(batches)


class LabelingTask(object):
    """Base class for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform and unbalanced label distribution).
    """

    def __init__(self, duration=3.2, batch_size=32, per_epoch=3600,
                 parallel=1, exhaustive=False):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel
        self.exhaustive = exhaustive

    def get_batch_generator(self, precomputed):
        """This method should be overriden by subclass

        Parameters
        ----------
        precomputed : `pyannote.audio.features.Precomputed`

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        return LabelingTaskGenerator(
            precomputed, duration=self.duration, per_epoch=self.per_epoch,
            batch_size=self.batch_size, parallel=self.parallel,
            exhaustive=self.exhaustive)

    @property
    def n_classes(self):
        """Number of classes"""
        msg = 'LabelingTask subclass must define `n_classes` property.'
        raise NotImplementedError(msg)

    def fit(self, model, feature_extraction, protocol,
            log_dir=None, subset='train', epochs=1000,
            restart=0, gpu=False):
        """Train model

        Parameters
        ----------
        model : torch.nn.Module
            Sequence labeling model.
        feature_extraction : pyannote.audio.features.Precomputed
            Precomputed features.
        protocol : pyannote.database.Protocol
            Evaluation protocol.
        subset : {'train', 'development', 'test'}, optional
            Subset to use for training. Defaults to "train".
        log_dir : str, optional
            Directory where models and other log files are stored.
            Defaults to not store anything.
        epochs : int, optional
            Train model for that many epochs. Defaults to 1000.
        restart : int, optional
            Restart training at this epoch. Defaults to train from scratch.
        gpu : bool, optional
            Use GPU. Defaults to CPU.

        Returns
        -------
        model : torch.nn.Module
            Trained model.
        """

        iterations = self.fit_iter(model, feature_extraction,
                                   protocol, log_dir=log_dir,
                                   subset=subset, epochs=epochs,
                                   restart=restart, gpu=gpu)

        for iteration in iterations:
            pass

        return iteration['model']

    def fit_iter(self, model, feature_extraction, protocol,
            log_dir=None, subset='train', epochs=1000,
            restart=0, gpu=False):
        """Same as `fit` except it returns an iterator

        Yields
        ------
        iteration : dict
            'epoch': <int>
            'model': <nn.Module>
            'loss': <float>
        """

        if log_dir is None and restart > 0:
            msg = ('One must provide `log_dir` when '
                   'using `restart` option.')
            raise ValueError(msg)

        log = log_dir is not None

        if model.n_classes != self.n_classes:
            raise ValueError('n_classes mismatch')
        n_classes = model.n_classes

        if log:
            checkpoint = Checkpoint(
                log_dir, restart=restart > 0)
            import tensorboardX
            writer = tensorboardX.SummaryWriter(
                log_dir=log_dir)

        self.batch_generator_ = self.get_batch_generator(feature_extraction)
        batches = self.batch_generator_(protocol, subset=subset)
        batch = next(batches)

        batches_per_epoch = self.batch_generator_.batches_per_epoch

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

            log_epoch = (epoch < 10) or (epoch % 5 == 0)
            log_epoch = log and log_epoch

            if log:
                loss_avg = 0.

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
                if log:
                    if gpu:
                        loss_ = float(loss.data.cpu().numpy())
                    else:
                        loss_ = float(loss.data.numpy())
                    loss_avg += loss_

                loss.backward()
                optimizer.step()

            if log:
                loss_avg /= batches_per_epoch
                writer.add_scalar('train/loss', loss_avg,
                                  global_step=epoch)

            if log_epoch:
                log_y_pred = np.hstack(log_y_pred)
                log_y_true = np.hstack(log_y_true)
                log_y_pred = log_y_pred.reshape((-1, n_classes))
                log_y_true = log_y_true.reshape((-1, ))
                for k in range(n_classes):
                    _, _, _, eer = det_curve(log_y_true == k,
                                             log_y_pred[:, k])
                    writer.add_scalar(f'train/estimate/eer/{k}',
                        eer, global_step=epoch)

            if log:
                checkpoint.on_epoch_end(epoch, model, optimizer)

            yield {'epoch': epoch, 'model': model, 'loss': loss}
