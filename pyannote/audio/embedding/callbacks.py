#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
import itertools
import datetime

from keras.callbacks import Callback
from keras.models import model_from_yaml
from pyannote.audio.keras_utils import CUSTOM_OBJECTS

from pyannote.audio.generators.labels import FixedDurationSequences
from pyannote.audio.generators.labels import VariableDurationSequences

from pyannote.audio.embedding.utils import pdist, cdist, l2_normalize, get_range
from pyannote.metrics.binary_classification import det_curve
from pyannote.metrics.plot.binary_classification import plot_det_curve
from pyannote.metrics.plot.binary_classification import plot_distributions

import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UpdateGeneratorEmbedding(Callback):

    def __init__(self, generator, extract_embedding, name='embedding'):
        super(UpdateGeneratorEmbedding, self).__init__()
        self.generator = generator
        self.extract_embedding = extract_embedding
        self.name = name

    def on_train_begin(self, logs={}):
        current_embedding = self.extract_embedding(self.model)
        architecture = model_from_yaml(
            current_embedding.to_yaml(),
            custom_objects=CUSTOM_OBJECTS)
        current_weights = current_embedding.get_weights()

        from pyannote.audio.embedding.base import SequenceEmbedding
        sequence_embedding = SequenceEmbedding()

        sequence_embedding.embedding_ = architecture
        sequence_embedding.embedding_.set_weights(current_weights)
        setattr(self.generator, self.name, sequence_embedding)

    def on_batch_end(self, batch, logs={}):
        current_embedding = self.extract_embedding(self.model)
        current_weights = current_embedding.get_weights()
        sequence_embedding = getattr(self.generator, self.name)
        sequence_embedding.embedding_.set_weights(current_weights)


class SpeakerDiarizationValidation(Callback):

    def __init__(self, glue, protocol, subset, log_dir):
        super(SpeakerDiarizationValidation, self).__init__()

        self.subset = subset
        self.distance = glue.distance
        self.extract_embedding = glue.extract_embedding
        self.log_dir = log_dir

        np.random.seed(1337)

        # initialize fixed duration sequence generator
        if glue.min_duration is None:
            # initialize fixed duration sequence generator
            generator = FixedDurationSequences(
                glue.feature_extractor,
                duration=glue.duration,
                step=glue.step,
                batch_size=-1)
        else:
            # initialize variable duration sequence generator
            generator = VariableDurationSequences(
                glue.feature_extractor,
                max_duration=glue.duration,
                min_duration=glue.min_duration,
                batch_size=-1)

        # randomly select (at most) 100 sequences from each label to ensure
        # all labels have (more or less) the same weight in the evaluation
        file_generator = getattr(protocol, subset)()
        X, y = zip(*generator(file_generator))
        X = np.vstack(X)
        y = np.hstack(y)
        unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
        n_labels = len(unique)
        indices = []
        for label in range(n_labels):
            i = np.random.choice(np.where(y == label)[0], size=min(100, counts[label]), replace=False)
            indices.append(i)
        indices = np.hstack(indices)
        X, y = X[indices], y[indices, np.newaxis]

        # precompute same/different groundtruth
        self.y_ = pdist(y, metric='chebyshev') < 1
        self.X_ = X

        self.EER_TEMPLATE_ = '{epoch:04d} {now} {eer:5f}\n'
        self.eer_ = []

    def on_epoch_end(self, epoch, logs={}):

        # keep track of current time
        now = datetime.datetime.now().isoformat()

        embedding = self.extract_embedding(self.model)
        fX = embedding.predict(self.X_)
        distance = pdist(fX, metric=self.distance)
        prefix = self.log_dir + '/{subset}.plot.{epoch:04d}'.format(
            subset=self.subset, epoch=epoch)

        # plot distance distribution every 20 epochs (and 10 first epochs)
        xlim = get_range(metric=self.distance)
        if (epoch < 10) or (epoch % 20 == 0):
            plot_distributions(self.y_, distance, prefix,
                               xlim=xlim, ymax=3, nbins=100, dpi=150)

        # plot DET curve once every 20 epochs (and 10 first epochs)
        if (epoch < 10) or (epoch % 20 == 0):
            eer = plot_det_curve(self.y_, distance, prefix,
                                 distances=True, dpi=150)
        else:
            _, _, _, eer = det_curve(self.y_, distance, distances=True)

        # store equal error rate in file
        mode = 'a' if epoch else 'w'
        path = self.log_dir + '/{subset}.eer.txt'.format(subset=self.subset)
        with open(path, mode=mode) as fp:
            fp.write(self.EER_TEMPLATE_.format(epoch=epoch, eer=eer, now=now))
            fp.flush()

        # plot eer = f(epoch)
        self.eer_.append(eer)
        best_epoch = np.argmin(self.eer_)
        best_value = np.min(self.eer_)
        fig = plt.figure()
        plt.plot(self.eer_, 'b')
        plt.plot([best_epoch], [best_value], 'bo')
        plt.plot([0, epoch], [best_value, best_value], 'k--')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('EER on {subset}'.format(subset=self.subset))
        TITLE = 'EER = {best_value:.5g} on {subset} @ epoch #{best_epoch:d}'
        title = TITLE.format(best_value=best_value,
                             best_epoch=best_epoch,
                             subset=self.subset)
        plt.title(title)
        plt.tight_layout()
        path = self.log_dir + '/{subset}.eer.png'.format(subset=self.subset)
        plt.savefig(path, dpi=150)
        plt.close(fig)


class SpeakerRecognitionValidation(Callback):

    def __init__(self, glue, protocol, subset, log_dir):
        super(SpeakerRecognitionValidation, self).__init__()
        self.glue = glue
        self.protocol = protocol
        self.subset = subset
        self.log_dir = log_dir

        self.EER_TEMPLATE_ = '{epoch:04d} {now} {eer:5f}\n'
        self.eer_ = []

    def on_epoch_end(self, epoch, logs={}):

        # keep track of current time
        now = datetime.datetime.now().isoformat()
        prefix = self.log_dir + '/{subset}.plot.{epoch:04d}'.format(
            epoch=epoch, subset=self.subset)

        from pyannote.audio.embedding.base import SequenceEmbedding
        sequence_embedding = SequenceEmbedding()
        sequence_embedding.embedding_ = self.glue.extract_embedding(self.model)

        from pyannote.audio.embedding.aggregation import \
            SequenceEmbeddingAggregation
        aggregation = SequenceEmbeddingAggregation(
            sequence_embedding,
            self.glue.feature_extractor,
            duration=self.glue.duration,
            min_duration=self.glue.min_duration,
            step=self.glue.step,
            internal=-2)

        # TODO / pass internal as parameter
        aggregation.cache_preprocessed_ = False

        # embed enroll and test recordings

        method = '{subset}_enroll'.format(subset=self.subset)
        enroll = getattr(self.protocol, method)(yield_name=True)

        method = '{subset}_test'.format(subset=self.subset)
        test = getattr(self.protocol, method)(yield_name=True)

        fX = {}
        for name, item in itertools.chain(enroll, test):
            if name in fX:
                continue
            embeddings = aggregation.apply(item)
            fX[name] = np.sum(embeddings.data, axis=0)

        # perform trials

        method = '{subset}_keys'.format(subset=self.subset)
        keys = getattr(self.protocol, method)()

        enroll_fX = l2_normalize(np.vstack([fX[name] for name in keys.index]))
        test_fX = l2_normalize(np.vstack([fX[name] for name in keys]))

        D = cdist(enroll_fX, test_fX, metric=self.glue.distance)

        y_true = []
        y_pred = []
        key_mapping = {0: None, -1: 0, 1: 1}
        for i, _ in enumerate(keys.index):
            for j, _ in enumerate(keys):
                y = key_mapping[keys.iloc[i, j]]
                if y is None:
                    continue

                y_true.append(y)
                y_pred.append(D[i, j])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # plot DET curve once every 20 epochs (and 10 first epochs)
        if (epoch < 10) or (epoch % 20 == 0):
            eer = plot_det_curve(y_true, y_pred, prefix,
                                 distances=True, dpi=150)
        else:
            _, _, _, eer = det_curve(y_true, y_pred, distances=True)

        # store equal error rate in file
        mode = 'a' if epoch else 'w'
        path = self.log_dir + '/{subset}.eer.txt'.format(subset=self.subset)
        with open(path, mode=mode) as fp:
            fp.write(self.EER_TEMPLATE_.format(epoch=epoch, eer=eer, now=now))
            fp.flush()

        # plot eer = f(epoch)
        self.eer_.append(eer)
        best_epoch = np.argmin(self.eer_)
        best_value = np.min(self.eer_)
        fig = plt.figure()
        plt.plot(self.eer_, 'b')
        plt.plot([best_epoch], [best_value], 'bo')
        plt.plot([0, epoch], [best_value, best_value], 'k--')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('EER on {subset}'.format(subset=self.subset))
        TITLE = 'EER = {best_value:.5g} on {subset} @ epoch #{best_epoch:d}'
        title = TITLE.format(best_value=best_value,
                             best_epoch=best_epoch,
                             subset=self.subset)
        plt.title(title)
        plt.tight_layout()
        path = self.log_dir + '/{subset}.eer.png'.format(subset=self.subset)
        plt.savefig(path, dpi=150)
        plt.close(fig)
