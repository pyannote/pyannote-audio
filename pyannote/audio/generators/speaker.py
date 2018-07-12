#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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


import warnings
import numpy as np
from pyannote.core import Segment
from pyannote.audio.features import Precomputed
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.batch import batchify, EndOfBatch
from pyannote.database import get_label_identifier
from pyannote.database import get_annotated


class SpeechSegmentGenerator(object):
    """Generate batch of pure speech segments with associated speaker labels

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    per_label : int, optional
        Number of speech turns per speaker in each batch. Defaults to 3.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep it all).
    per_fold : int, optional
        Number of speakers in each batch. Defaults to all speakers.
    duration : float, optional
        Use fixed duration segments with this `duration`.
        Defaults (None) to using variable duration segments.
    min_duration : float, optional
        In case `duration` is None, set segment minimum duration.
    max_duration : float, optional
        In case `duration` is None, set segment maximum duration.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, feature_extraction,
                 per_label=3, per_fold=None,
                 duration=None, min_duration=None, max_duration=None,
                 label_min_duration=0., parallel=1):

        super(SpeechSegmentGenerator, self).__init__()

        self.feature_extraction = feature_extraction
        self.per_label = per_label
        self.per_fold = per_fold
        self.duration = duration
        self.parallel = parallel
        self.label_min_duration = label_min_duration

        if self.duration is None:
            self.min_duration = min_duration
            self.max_duration = max_duration
        else:
            self.min_duration = self.duration
            self.max_duration = self.duration

        self.min_duration_ = 0. if self.min_duration is None \
                                else self.min_duration

        self.weighted_ = True

    def initialize(self, protocol, subset='train'):

        self.data_ = {}
        databases = set()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of database
            database = current_file['database']
            databases.add(database)

            # get annotation for current file
            annotation = current_file['annotation']

            # pre-load features.
            if isinstance(self.feature_extraction, Precomputed) and \
               not self.feature_extraction.use_memmap:
                msg = ('Loading all precomputed features in memory. '
                       'Set "use_memmap" to True if you run out of memory.')
                warnings.warn(msg)

            # loop on each label in current file
            for label in annotation.labels():

                # get all segments with current label
                timeline = annotation.label_timeline(label)

                # remove segments shorter than min_duration (when provided)
                segments = [s for s in timeline
                              if s.duration > self.min_duration_]

                # corner case where no segment is long enough
                # and we removed them all...
                if not segments:
                    continue

                # total duration of label in current_file (after removal of
                # short segments).
                duration = sum(s.duration for s in segments)

                # store all these in data_ dictionary
                # datum = (segment_generator, duration, current_file, features)
                datum = (segments, duration, current_file)
                l = get_label_identifier(label, current_file)
                self.data_.setdefault(l, []).append(datum)

        # remove labels with less than 'label_min_duration' of speech
        # otherwise those may generate the same segments over and over again
        dropped_labels = set()
        for label, data in self.data_.items():
            total_duration = sum(datum[1] for datum in data)
            if total_duration < self.label_min_duration:
                dropped_labels.add(label)

        for label in dropped_labels:
            self.data_.pop(label)

        self.labels_ = {label: i for i, label in enumerate(self.data_)}

        self.domains_ = {}
        self.domains_['database'] = {db: i for i, db in enumerate(databases)}

    def generator(self):

        labels = list(self.data_)

        while True:

            # shuffle labels
            np.random.shuffle(labels)

            # loop on each label
            for label in labels:

                # load data for this label
                # segment_generators, durations, files, features = \
                #     zip(*self.data_[label])
                segments, durations, files = zip(*self.data_[label])

                # choose 'per_label' files at random with probability
                # proportional to the total duration of 'label' in those files
                probabilities = durations / np.sum(durations)
                chosen = np.random.choice(len(files), size=self.per_label,
                                          p=probabilities)

                # loop on (randomly) chosen files
                for i in chosen:

                    # choose one segment at random with
                    # probability proportional to duration
                    # segment = next(segment_generators[i])
                    segment = next(
                        random_segment(segments[i], weighted=self.weighted_))

                    if self.duration is None:

                        if self.min_duration is None:

                            # case: no duration | no min | no max
                            # keep segment as it is
                            if self.max_duration is None:
                                sub_segment = segment

                            # case: no duration | no min | max
                            else:

                                # if segment is too long, choose sub-segment
                                # at random at exactly max_duration
                                if segment.duration > self.max_duration:
                                    sub_segment = next(random_subsegment(
                                        segment, self.max_duration))

                                # otherwise, keep segment as it is
                                else:
                                    sub_segment = segment

                        else:
                            # case: no duration | min | no max
                            # keep segment as it is (too short segments have
                            # already been filtered out)
                            if self.max_duration is None:
                                sub_segment = segment

                            # case: no duration | min | max
                            else:
                                # choose sub-segment at random between
                                # min_duration and max_duration
                                sub_segment = next(random_subsegment(
                                    segment, self.max_duration,
                                    min_duration=self.min_duration))

                        X = self.feature_extraction.crop(
                            files[i], sub_segment, mode='center')

                    else:

                        # choose sub-segment at random at exactly duration
                        sub_segment = next(random_subsegment(
                            segment, self.duration))

                        X = self.feature_extraction.crop(
                            files[i], sub_segment, mode='center',
                            fixed=self.duration)

                    database = files[i]['database']
                    extra = {'label': label,
                             'database': database}

                    yield {'X': X,
                           'y': self.labels_[label],
                           'y_database': self.domains_['database'][database],
                           'extra': extra}

    @property
    def batch_size(self):
        if self.per_fold is not None:
            return self.per_label * self.per_fold
        return self.per_label * len(self.data_)

    @property
    def batches_per_epoch(self):

        # one minute per speaker
        duration_per_epoch = 60 * len(self.data_)

        # number of segments per batch
        if self.per_fold is None:
            segments_per_batch = self.per_label * len(self.data_)
        else:
            segments_per_batch = self.per_label * self.per_fold

        # duration per batch
        if self.duration is None:
            if self.min_duration is None:
                duration_per_batch = self.max_duration * segments_per_batch
            else:
                duration_per_batch = self.min_duration * segments_per_batch
        else:
            duration_per_batch = self.duration * segments_per_batch

        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def n_classes(self):
        return len(self.data_)

    @property
    def labels(self):
        labels, _ = zip(*sorted(self.labels_.items(),
                                key=lambda x: x[1]))
        return labels

    @property
    def signature(self):
        return {'X': {'@': (None, None)},
                'y': {'@': (None, np.stack)},
                'y_database': {'@': (None, np.stack)},
                'extra': {'label': {'@': (None, None)},
                         'database': {'@': (None, None)}}}

    def __call__(self, protocol, subset='train'):

        self.initialize(protocol, subset=subset)

        batch_size = self.batch_size
        batches_per_epoch = self.batches_per_epoch

        generators = []
        if self.parallel:

            for i in range(self.parallel):
                generator = self.generator()
                batches = batchify(generator, self.signature,
                                   batch_size=batch_size,
                                   prefetch=batches_per_epoch)
                generators.append(batches)
        else:
            generator = self.generator()
            batches = batchify(generator, self.signature,
                               batch_size=batch_size, prefetch=0)
            generators.append(batches)

        while True:
            # get `batches_per_epoch` batches from each generator
            for batches in generators:
                for _ in range(batches_per_epoch):
                    yield next(batches)


class SpeechTurnSubSegmentGenerator(SpeechSegmentGenerator):
    """Generates batches of speech turn fixed-duration sub-segments

    Usage
    -----
    >>> generator = SpeechTurnSubSegmentGenerator(precomputed, 3.)
    >>> batches = generator(protocol)
    >>> batch = next(batches)

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    duration : float
        Fixed segment duration.
    per_turn : int, optional
        Number of segments per speech turn. Defaults to 10.
        For short speech turns, a heuristic adapts this number to reduce the
        number of overlapping segments.
    per_label : int, optional
        Number of speech turns per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        Number of speakers in each batch. Defaults to all speakers.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, feature_extraction, duration, per_label=3,
                 per_fold=None, per_turn=10, parallel=1):

        super(SpeechTurnSubSegmentGenerator, self).__init__(
            feature_extraction, per_label=per_label, per_fold=per_fold,
            duration=None, min_duration=duration, max_duration=None)

        # this is to make sure speech turns are selected at random
        self.weighted_ = False
        self.duration_ = duration
        self.per_turn = per_turn

        # estimate number of samples in each subsequence
        sw = feature_extraction.sliding_window
        ranges = sw.crop(Segment(0, self.duration_), mode='center',
                         fixed=self.duration_, return_ranges=True)
        self.n_samples_ = ranges[0][1] - ranges[0][0]

    def iter_segments_(self, X):
        """Generate fixed length sub-segments of X

        Parameters
        ----------
        X : np.array (n_samples, dimension)
            Speech turn features.

        Yields
        ------
        x : np.array (self.n_samples_, dimension)
            Sub-segment features.
        """

        # heuristic that tries to avoid highly-overlapping sub-segments
        # (i.e. with more than 50% overlap on average) for short speech turns
        n_samples = len(X)
        n = (n_samples - self.n_samples_) // (self.n_samples_ // 2) + 1
        n = min(n, self.per_turn)

        # choose (and yield) n sub-segments at random
        for i in np.random.randint(0, n_samples - self.n_samples_, n):
            yield X[i: i + self.n_samples_]

    def generator(self):
        """Generate speech turn fixed-length sub-segments"""

        # generates speech turns long enough to contain at least one segment
        speech_turns = super(SpeechTurnSubSegmentGenerator, self).generator()

        # number of speech turns per "speech turn batch"
        if self.per_fold is not None:
            n_speech_turns = self.per_label * self.per_fold
        else:
            n_speech_turns = self.per_label * len(self.data_)

        endOfBatch = EndOfBatch()
        while True:

            # for each speech turn in batch
            for z in range(n_speech_turns):
                speech_turn = next(speech_turns)

                # for each segment in speech turn
                for X in self.iter_segments_(speech_turn['X']):

                    # all but 'X' fields are left unchanged
                    segment = dict(speech_turn)
                    segment['X'] = X

                    # remember that this segment belongs to this speech turn
                    segment['z'] = z

                    yield segment

            # let `batchify` know that the "segment batch" is complete
            yield endOfBatch

    @property
    def batch_size(self):
        return -1

    @property
    def signature(self):
        return {'X': {'@': (None, None)},
                'y': {'@': (None, np.stack)},
                'z': {'@': (None, np.stack)},
                'y_database': {'@': (None, np.stack)},
                'extra': {
                    'label': {'@': (None, None)},
                    'database': {'@': (None, None)}}}

    @property
    def batches_per_epoch(self):

        # one minute per speaker
        duration_per_epoch = 60 * len(self.data_)

        # number of segments per batch
        if self.per_fold is None:
            segments_per_batch = self.per_label * len(self.data_) * self.per_turn
        else:
            segments_per_batch = self.per_label * self.per_fold * self.per_turn

        duration_per_batch = self.duration_ * segments_per_batch

        return int(np.ceil(duration_per_epoch / duration_per_batch))
