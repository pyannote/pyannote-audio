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
import datetime

from keras.callbacks import Callback
from keras.models import model_from_yaml
from pyannote.audio.keras_utils import CUSTOM_OBJECTS

from pyannote.audio.embedding.base import SequenceEmbedding

from pyannote.audio.generators.labels import FixedDurationSequences
from pyannote.audio.generators.labels import VariableDurationSequences

from scipy.spatial.distance import pdist
from pyannote.metrics.plot.binary_classification import plot_det_curve
from pyannote.metrics.plot.binary_classification import plot_distributions

class UpdateGeneratorEmbedding(Callback):

    def __init__(self, generator, extract_embedding, name='embedding'):
        super(UpdateGeneratorEmbedding, self).__init__()
        self.generator = generator
        self.extract_embedding = extract_embedding
        self.name = name

    def _copy_embedding(self, current_model):

        # make a copy of current embedding
        embedding = self.extract_embedding(current_model)
        embedding_copy = model_from_yaml(
            embedding.to_yaml(), custom_objects=CUSTOM_OBJECTS)
        embedding_copy.set_weights(embedding.get_weights())

        # encapsulate it in a SequenceEmbedding instance
        sequence_embedding = SequenceEmbedding()
        sequence_embedding.embedding_ = embedding_copy
        return sequence_embedding

    def on_train_begin(self, logs={}):
        embedding = self._copy_embedding(self.model)
        setattr(self.generator, self.name, embedding)

    def on_batch_begin(self, batch, logs={}):
        embedding = self._copy_embedding(self.model)
        setattr(self.generator, self.name, embedding)


class ValidateEmbedding(Callback):

    def __init__(self, glue, file_generator, log_dir):
        super(ValidateEmbedding, self).__init__()

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
                step=glue.duration,
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
        if self.distance == 'angular':
            cosine_distance = pdist(fX, metric='cosine')
            distances = np.arccos(np.clip(1.0 - cosine_distance, -1.0, 1.0))
        else:
            distances = pdist(fX, metric=metric)

        prefix = self.log_dir + '/plot.{epoch:04d}'.format(epoch=epoch)

        # plot distributions of positive & negative scores
        if self.distance == 'angular':
            xlim = (0, np.pi)
        elif self.distance == 'sqeuclidean':
            xlim = (0, 4)
        elif self.distance == 'cosine':
            xlim = (-1.0, 1.0)
        plot_distributions(self.y_, distances, prefix,
                           xlim=xlim, ymax=3, nbins=100)

        # plot DET curve
        eer = plot_det_curve(self.y_, -distances, prefix)

        # store equal error rate in file
        mode = 'a' if epoch else 'w'
        with open(self.log_dir + '/eer.txt', mode=mode) as fp:
            fp.write(self.EER_TEMPLATE_.format(epoch=epoch, eer=eer, now=now))
            fp.flush()
