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
from pyannote.audio.train.task import Task, TaskType, TaskOutput


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class OverlapDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training overlap detection

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models
        that include the feature extraction step (e.g. SincNet) and
        therefore use a different cropping mode. Defaults to 'center'.
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

    def __init__(self, feature_extraction, protocol, subset='train',
                 resolution=None, alignment=None, duration=3.2,
                 snr_min=0, snr_max=10,
                 batch_size=32, per_epoch=1, parallel=1):

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.raw_audio_ = RawAudio(sample_rate=feature_extraction.sample_rate)

        super().__init__(feature_extraction, protocol, subset=subset,
                         resolution=resolution, alignment=alignment,
                         duration=duration,
                         batch_size=batch_size, per_epoch=per_epoch,
                         parallel=parallel, shuffle=True)

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

            # choose random subsegment
            # duration = np.random.rand() * self.duration
            sequence = next(random_subsegment(segment, self.duration))

            # get corresponding waveform
            X = self.raw_audio_.crop(current_file,
                                     sequence,
                                     mode='center',
                                     fixed=self.duration)

            # get corresponding labels
            y = datum['y'].crop(sequence,
                                mode=self.alignment,
                                fixed=self.duration)

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

            # shuffle files
            np.random.shuffle(uris)

            # loop on shuffled files
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

                    y = datum['y'].crop(sequence, mode=self.alignment,
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

    def __call__(self):
        """(Parallelized) batch generator"""

        # number of batches needed to complete an epoch
        batches_per_epoch = self.batches_per_epoch

        def generator():

            sliding_samples = self.sliding_samples()
            overlap_samples = self.overlap_samples()

            while True:

                # get fixed duration random sequence
                original = next(sliding_samples)

                if np.random.rand() < 0.5:
                    pass

                else:
                    # get random overlapping sequence
                    overlap = next(overlap_samples)

                    # select SNR at random
                    snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
                    alpha = np.exp(-np.log(10) * snr / 20)

                    original['waveform'] += alpha * overlap['waveform']
                    original['y'] += overlap['y']

                speaker_count = np.sum(original['y'], axis=1, keepdims=True)
                original['y'] = np.int64(speaker_count > 1)

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

    @property
    def specifications(self):
        return {
            'task': Task(type=TaskType.MULTI_CLASS_CLASSIFICATION,
                         output=TaskOutput.SEQUENCE),
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': ['non_overlap', 'overlap']},
        }


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
    """

    def get_batch_generator(self, feature_extraction, protocol, subset='train',
                            resolution=None, alignment=None):
        """
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return OverlapDetectionGenerator(
            feature_extraction,
            protocol, subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            parallel=self.parallel)
