#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
from pyannote.database import get_annotated
from pyannote.core import Segment
from pyannote.core import Timeline

from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import batchify

from pyannote.audio.features import RawAudio

from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_CLASSIFICATION


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class OverlapDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training overlap detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    snr_min, snr_max : int, optional
        Defines Signal-to-Overlap Ratio range in dB. Defaults to [0, 10].
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

    def __init__(self, feature_extraction, duration=3.2, snr_min=0, snr_max=10,
                 batch_size=32, per_epoch=1, parallel=1):

        super().__init__(feature_extraction, duration=duration,
                         batch_size=batch_size, per_epoch=per_epoch,
                         parallel=parallel, shuffle=True)

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.raw_audio_ = RawAudio(
            sample_rate=self.feature_extraction.sample_rate)

    def overlap_samples(self):
        """Random overlap samples

        Returns
        -------
        samples : generator
            Generator that yields {'waveform': ..., 'y': ...} samples
            indefinitely.
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

            # choose random duration subsegment at random
            duration = np.random.rand() * self.duration
            sequence = next(random_subsegment(segment, duration))

            # get corresponding waveform
            X = self.raw_audio_.crop(current_file, sequence,
                                     mode='center', fixed=duration)

            # get corresponding labels
            y = datum['y'].crop(sequence, mode='center', fixed=duration)

            yield {'waveform': normalize(X),
                   'y': y}

    def sliding_samples(self):
        """Sliding window

        Returns
        -------
        samples : generator
            Generator that yields {'waveform': ..., 'y': ...} samples
            indefinitely.
        """

        uris = list(self.data_)
        durations = np.array([self.data_[uri]['duration'] for uri in uris])
        probabilities = durations / np.sum(durations)

        sliding_segments = SlidingSegments(duration=self.duration,
                                           step=self.duration,
                                           source='annotated')

        while True:

            np.random.shuffle(uris)

            # loop on all files
            for uri in uris:

                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum['current_file'])

                # read waveform for the whole file
                waveform = self.raw_audio_(current_file)

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                shifted_segments = [
                    Segment(s.start + np.random.random() * self.duration, s.end)
                    for s in get_annotated(current_file)]
                # deal with corner case where a shifted segment would be empty
                shifted_segments = [s for s in shifted_segments if s]
                annotated = Timeline(segments=shifted_segments)
                current_file['annotated'] = annotated

                if self.shuffle:
                    samples = []

                for sequence in sliding_segments.from_file(current_file):

                    X = waveform.crop(sequence, mode='center',
                                      fixed=self.duration)

                    y = datum['y'].crop(sequence, mode='center',
                                        fixed=self.duration)

                    sample = {'waveform': normalize(X),
                              'y': y}

                    if self.shuffle:
                        samples.append(sample)
                    else:
                        yield sample

                if self.shuffle:
                    np.random.shuffle(samples)
                    for sample in samples:
                        yield sample

    @property
    def signature(self):
        return {'X': {'@': (None, np.stack)},
                'y': {'@': (None, np.stack)}}

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * 24 * 60 * 60
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def labels(self):
        return list(self.labels_)

    def __call__(self, protocol, subset='train'):
        """(Parallelized) batch generator"""

        # pre-load useful information about protocol once and for all
        self.initialize(protocol, subset=subset)

        # number of batches needed to complete an epoch
        batches_per_epoch = self.batches_per_epoch

        def generator():

            sliding_samples = self.sliding_samples()
            overlap_samples = self.overlap_samples()

            while True:

                # get original sequence
                original = next(sliding_samples)

                n_samples = len(original['waveform'])
                n_labels = len(original['y'])

                # get sample to overlap
                overlap = next(overlap_samples)
                n_overlap = len(overlap['waveform'])

                # randomly choose were to add overlap
                i = np.random.randint(2*n_samples) - n_samples
                l = max(i, 0)
                r = min(max(i + n_overlap, 0), n_samples)

                # select SNR at random
                snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
                alpha = np.exp(-np.log(10) * snr / 20)

                # add overlap
                original['waveform'][l:r] += alpha * overlap['waveform'][:r-l]

                # update "who speaks when" labels
                l = int(l * n_labels / n_samples)
                r = int(r * n_labels / n_samples)

                if r-l > len(overlap['y']):
                    r = r-1
                original['y'][l:r] += overlap['y'][:r-l]

                speaker_count = np.sum(original['y'], axis=1, keepdims=True)
                original['y'] = np.int64(speaker_count > 1)

                # run feature extraction
                #original['X'] = self.feature_extraction(original).data

                # run feature extraction
                original['duration'] = self.duration
                original['X'] = self.feature_extraction.crop(
                    original, Segment(0, self.duration), mode='center',
                    fixed=self.duration)

                del original['waveform']
                del original['duration']

                yield original


        generators = []

        if self.parallel:
            for _ in range(self.parallel):

                # initialize one sample generator
                samples = generator()

                # batchify it and make sure at least
                # `batches_per_epoch` batches are prefetched.
                batches = batchify(samples, self.signature,
                                   batch_size=self.batch_size,
                                   prefetch=batches_per_epoch)

                # add batch generator to the list of (background) generators
                generators.append(batches)
        else:

            # initialize one sample generator
            samples = generator()

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


class OverlapDetection(LabelingTask):
    """Train overlap detection

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

    Usage
    -----
    >>> task = OverlapDetection()

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(precomputed.dimension, task.n_classes)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass

    """

    def __init__(self, **kwargs):
        super(OverlapDetection, self).__init__(**kwargs)

    def get_batch_generator(self, precomputed):
        return OverlapDetectionGenerator(
            precomputed, duration=self.duration,
            per_epoch=self.per_epoch, batch_size=self.batch_size,
            parallel=self.parallel)

    @property
    def task_type(self):
        return TASK_CLASSIFICATION

    @property
    def n_classes(self):
        return 2
