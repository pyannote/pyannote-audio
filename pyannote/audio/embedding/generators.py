#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.batch import batchify
from ..models import TASK_REPRESENTATION_LEARNING
from pyannote.audio.features import RawAudio


class SpeechSegmentGenerator(object):
    """Generate batch of pure speech segments with associated speaker labels

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    per_label : int, optional
        Number of speech turns per speaker in each batch. Defaults to 3.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep it all).
    per_fold : int, optional
        Number of speakers in each batch. Defaults to all speakers.
    per_epoch : float, optional
        Number of days per epoch. Defaults to 7 (a week).
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
    in_memory : `bool`, optional
        Pre-load training set in memory.

    """

    def __init__(self, feature_extraction, protocol, subset='train',
                 per_label=3, per_fold=None, per_epoch=7,
                 duration=None, min_duration=None, max_duration=None,
                 label_min_duration=0., parallel=1, in_memory=False):

        super(SpeechSegmentGenerator, self).__init__()

        self.feature_extraction = feature_extraction
        self.per_label = per_label
        self.per_fold = per_fold
        self.per_epoch = per_epoch
        self.duration = duration
        self.parallel = parallel
        self.label_min_duration = label_min_duration

        self.in_memory = in_memory
        if self.in_memory:
            if not isinstance(feature_extraction, RawAudio):
                msg = (
                    f'"in_memory" option is only supported when '
                    f'working from the waveform.'
                )
                raise ValueError(msg)

        if self.duration is None:
            self.min_duration = min_duration
            self.max_duration = max_duration
        else:
            self.min_duration = self.duration
            self.max_duration = self.duration

        self.min_duration_ = 0. if self.min_duration is None \
                                else self.min_duration

        self.weighted_ = True

        self._load_metadata(protocol, subset=subset)

    def _load_metadata(self, protocol, subset='train'):

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of unique file labels
            for key, value in current_file.items():
                if key in ['annotation', 'annotated']:
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(value)

            # get annotation for current file
            annotation = current_file['annotation']

            if self.in_memory:
                current_file['waveform'] = \
                    self.feature_extraction(current_file).data

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
                self.data_.setdefault(label, []).append(datum)

        # remove labels with less than 'label_min_duration' of speech
        # otherwise those may generate the same segments over and over again
        dropped_labels = set()
        for label, data in self.data_.items():
            total_duration = sum(datum[1] for datum in data)
            if total_duration < self.label_min_duration:
                dropped_labels.add(label)

        for label in dropped_labels:
            self.data_.pop(label)

        self.file_labels_ = {k: sorted(file_labels[k]) for k in file_labels}
        self.segment_labels_ = sorted(self.data_)

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

                    yield {'X': X, 'y': self.segment_labels_.index(label)}

    @property
    def batch_size(self):
        if self.per_fold is not None:
            return self.per_label * self.per_fold
        return self.per_label * len(self.data_)

    @property
    def batches_per_epoch(self):

        # duration per epoch
        duration_per_epoch = self.per_epoch * 24 * 60 * 60

        # (average) duration per segment
        if self.duration is None:
            min_duration = 0. if self.min_duration is None \
                              else self.min_duration
            duration = .5 * (min_duration + self.max_duration)
        else:
            duration = self.duration

        # (average) duration per batch
        duration_per_batch = duration * self.batch_size

        # number of batches per epoch
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def signature(self):
        return {
            'X': {'@': (None, None)},
            'y': {'@': (None, np.stack)},
        }

    @property
    def specifications(self):
        return {
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.segment_labels_},
            'task': TASK_REPRESENTATION_LEARNING,
        }

    def __call__(self):

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
