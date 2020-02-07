#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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

from typing import Optional
import itertools
import numpy as np
from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from pyannote.audio.features import RawAudio
from ..train.generator import BatchGenerator


class SpeechSegmentGenerator(BatchGenerator):
    """Generate batch of pure speech segments with associated speaker labels

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    protocol : `pyannote.database.Protocol`
    subset : {'train', 'development', 'test'}
    duration : float, optional
        Chunks duration, in seconds. Defaults to 1.
    per_turn : int, optional
        Number of chunks per speech turn. Defaults to 1.
    per_label : int, optional
        Number of speech turns per speaker in each batch.
        Defaults to 3.
    per_fold : int, optional
        Number of different speakers in each batch.
        Defaults to all speakers.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    label_min_duration : float, optional
        Remove speakers with less than `label_min_duration` seconds of speech.
        Defaults to 0 (i.e. keep it all).
    """

    def __init__(self, feature_extraction,
                       protocol,
                       subset='train',
                       duration: float = 1.,
                       per_turn: int = 1,
                       per_label: int = 3,
                       per_fold: Optional[int] = None,
                       per_epoch: float = None,
                       label_min_duration: float = 0.):

        self.feature_extraction = feature_extraction
        self.per_turn = per_turn
        self.per_label = per_label
        self.per_fold = per_fold
        self.duration = duration
        self.label_min_duration = label_min_duration
        self.weighted_ = True

        total_duration = self._load_metadata(protocol, subset=subset)
        if per_epoch is None:
            per_epoch = total_duration / (24 * 60 * 60)
        self.per_epoch = per_epoch

    def _load_metadata(self, protocol, subset='train') -> float:
        """Load training set metadata

        This function is called once at instantiation time, returns the total
        training set duration, and populates the following attributes:

        Attributes
        ----------
        data_ : dict
            Dictionary where keys are speaker labels and values are lists of
            (segments, duration, current_file) tuples where
            - segments is a list of segments by the speaker in the file
            - duration is total duration of speech by the speaker in the file
            - current_file is the file (as ProtocolFile)

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file-level labels in protocol

        Returns
        -------
        duration : float
            Total duration of annotated segments, in seconds.
        """

        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of unique file labels
            for key in current_file:
                if key in ['annotation', 'annotated', 'audio', 'duration']:
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(current_file[key])

            # get annotation for current file
            annotation = current_file['annotation']

            # loop on each label in current file
            for label in annotation.labels():

                # get all segments with current label
                timeline = annotation.label_timeline(label)

                # remove segments shorter than 'duration'
                segments = [s for s in timeline
                              if s.duration > self.duration]

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

        return sum(sum(datum[1] for datum in data)
                   for data in self.data_.values())

    def samples(self):

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

                    # choose per_turn chunk(s) at random
                    for chunk in itertools.islice(
                        random_subsegment(segment, self.duration),
                        self.per_turn):

                        yield {
                            'X': self.feature_extraction.crop(
                                files[i], chunk, mode='center',
                                fixed=self.duration),
                            'y': self.segment_labels_.index(label),
                        }

    @property
    def batch_size(self):
        if self.per_fold is not None:
            return self.per_turn * self.per_label * self.per_fold
        return self.per_turn * self.per_label * len(self.data_)

    @property
    def batches_per_epoch(self):

        # duration per epoch
        duration_per_epoch = self.per_epoch * 24 * 60 * 60

        # (average) duration per batch
        duration_per_batch = self.duration * self.batch_size

        # number of batches per epoch
        return int(np.ceil(duration_per_epoch / duration_per_batch))

    @property
    def specifications(self):
        return {
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': self.segment_labels_},
            'task': Task(type=TaskType.REPRESENTATION_LEARNING,
                         output=TaskOutput.VECTOR),
        }
