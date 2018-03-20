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


from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.batch import batchify

from pyannote.database import get_annotated
from pyannote.database import get_unique_identifier

import numpy as np
import torch.nn as nn
from .base import LabelingTask


class SpeechActivityDetectionGenerator(object):
    """Generate batch of segments with associated frame-wise labels

    Parameters
    ----------
    precomputed : pyannote.audio.features.Precomputed
        Precomputed features
    duration : float, optional
        Use fixed duration segments with this `duration`.
    per_epoch : float, optional
        Total audio duration per epoch, in seconds.
        Defaults to one hour (3600).
    batch_size : int, optional
        Batch size. Defaults to 32.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, precomputed, duration=3.2, batch_size=32,
                 per_epoch=3600, parallel=1):

        super(SpeechActivityDetectionGenerator, self).__init__()

        self.precomputed = precomputed
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel

    def initialize(self, protocol, subset='train'):

        self.data_ = {}
        databases = set()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of database
            database = current_file['database']
            databases.add(database)

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
            # datum = (segment_generator, duration, current_file, features)
            datum = {'segments': segments,
                     'duration': duration,
                     'current_file': current_file}
            uri = get_unique_identifier(current_file)
            self.data_[uri] = datum

    def fill_y(self, y, sequence, current_file):

        n_samples = len(y)
        sw = self.precomputed.sliding_window()
        left, _ = sw.crop(sequence, mode='center', return_ranges=True)[0]

        turns = current_file['annotation']
        turns = turns.crop(sequence).get_timeline().support()
        for speech in turns:
            l, r = sw.crop(speech, mode='center', return_ranges=True)[0]
            l = max(0, l - left)
            r = min(r - left, n_samples)
            y[l:r] = 1

        return

    def generator(self):

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

            n_samples, _ = X.shape
            y = np.zeros((n_samples, ), dtype=int)
            self.fill_y(y, sequence, current_file)

            yield {'X': X, 'y': y}

    @property
    def signature(self):
        return {'X': {'type': 'ndarray'},
                'y': {'type': 'ndarray'}}

    @property
    def batches_per_epoch(self):
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(self.per_epoch / duration_per_batch))

    def __call__(self, protocol, subset='train'):

        self.initialize(protocol, subset=subset)

        batch_size = self.batch_size
        signature = self.signature
        batches_per_epoch = self.batches_per_epoch

        generators = []
        if self.parallel:

            for i in range(self.parallel):
                generator = self.generator()
                batches = batchify(generator, signature, batch_size=batch_size,
                                   prefetch=batches_per_epoch)
                generators.append(batches)
        else:
            generator = self.generator()
            batches = batchify(generator, signature, batch_size=batch_size,
                               prefetch=0)
            generators.append(batches)

        while True:
            # get `batches_per_epoch` batches from each generator
            for batches in generators:
                for _ in range(batches_per_epoch):
                    yield next(batches)


class SpeechActivityDetection(LabelingTask):

    def __init__(self, duration=3.2, batch_size=32, parallel=1):
        super(SpeechActivityDetection, self).__init__(duration=duration,
                                                      batch_size=batch_size,
                                                      parallel=parallel)

    def get_batch_generator(self, precomputed):
        return SpeechActivityDetectionGenerator(
            precomputed, duration=self.duration,
            batch_size=self.batch_size, parallel=self.parallel)

    @property
    def n_classes(self):
        return 2
