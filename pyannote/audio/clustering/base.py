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

from typing import Optional
from pathlib import Path
import numpy as np
from collections import Counter

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import Timeline
from pyannote.core import Annotation

from pyannote.core.utils.numpy import one_hot_encoding

from pyannote.database.protocol.protocol import Protocol
from pyannote.database.protocol import SpeakerDiarizationProtocol

from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.batch import batchify

from pyannote.audio.models import TASK_MULTI_CLASS_CLASSIFICATION
from pyannote.audio.features import Precomputed


class ConversationGenerator:
    """Fake conversation generator

    Parameters
    ----------
    protocol : `pyannote.database.protocol.SpeakerDiarizationProtocol`
    subset : {'train', 'development', 'test'}, optional
        Defaults to 'train'.
    duration : `float`, optional
        When set, make sure conversations are exactly `duration` seconds long.
    window : `SlidingWindow`, optional
        When set, one-hot encode conversation using this sliding window.

    Usage
    -----
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('AMI.SpeakerDiarization.MixHeadset')
    >>> conversations = ConversationGenerator(protocol)()
    >>> conversation = next(conversations)
    """

    def __init__(self, protocol: SpeakerDiarizationProtocol,
                       subset: Optional[str] = 'train',
                       duration: Optional[float] = 180.0,
                       window: Optional[SlidingWindow] = None):

        self.protocol = protocol
        self.subset = subset
        self.duration = duration
        self.window = window

        self.support_ = Timeline(segments=[Segment(0, self.duration)])
        self._load_metadata()

    def _load_metadata(self):

        self.files_ = []
        max_speakers = 0
        for current_file in getattr(self.protocol, self.subset)():

            # rename speakers as 0, 1, 2, 3, etc...
            current_file['annotation'].rename_labels(generator='int',
                                                     copy=False)
            self.files_.append(current_file)
            max_speakers = max(max_speakers,
                               len(current_file['annotation'].labels()))

        self.max_speakers_ = max_speakers

    def _delexicalize(self, y):
        """Make sure clusters are numbered in chronological order"""
        _, index, inverse = np.unique(y,
                                      return_index=True,
                                      return_inverse=True)
        return np.argsort(np.argsort(index))[inverse]

    def _encode(self, conversation):
        y, _ = one_hot_encoding(conversation, self.support_,
                                self.window, mode='center')
        return self._delexicalize(np.argmax(y.data, axis=1))

    def __call__(self):

        conversation = Annotation(uri='fake')
        current_t = 0.

        while True:

            np.random.shuffle(self.files_)
            for current_file in self.files_:

                # select one chunk (approx self.duration long) from file
                region = next(random_segment(current_file['annotated'],
                                             weighted=True))
                chunk = next(random_subsegment(
                    region, min(self.duration, region.duration)))

                chunk = current_file['annotation'].crop(chunk, mode='loose')
                for s, _, label in chunk.itertracks(yield_label=True):
                    # TODO. modify s.duration at random
                    # TODO. modify label at random
                    segment = Segment(current_t, current_t + s.duration)
                    conversation[segment] = label
                    current_t += s.duration

                if current_t > self.duration:

                    conversation = conversation.crop(
                        self.support_, mode='intersection').support()

                    if self.window is None:
                        yield conversation
                    else:
                        yield self._encode(conversation)

                    conversation = Annotation(uri='fake')
                    current_t = 0.


class ClusteringBatchGenerator:
    """Generate batches for training clustering

    Parameters
    ----------
    embedding : `Path`
        Path to precomputed embeddings.
    conversation_protocol : `pyannote.database.protocol.SpeakerDiarizationProtocol`
    conversation_subset : {'train', 'development', 'test'}, optional
        Protocol and subset used as conversation database.
    conversation_duration : `float`, optional
        Defaults to 180.
    speaker_protocol : `pyannote.database.protocol.protocol.Protocol`
    speaker_subset : {'train', 'development', 'test'}, optional
        Protocol and subset used as speaker database. Default to
        `conversation_protocol` and `conversation_subset` when
        `speaker_protocol` is not provided.
    speaker_min_duration : `float`, optional
        Remove speakers with less than `speaker_min_duration` seconds of
        speech. Defaults to 0 (i.e. keep them all).
    batch_size : `int`, optional
        Batch size. Defaults to 1.
    per_epoch : `float`, optional
        Number of days per epoch. Defaults to 30 (a month).
    parallel : `int`, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, embedding: Path,
                       conversation_protocol: SpeakerDiarizationProtocol,
                       conversation_subset: Optional[str] = 'train',
                       conversation_duration: Optional[float] = 180,
                       speaker_protocol: Optional[Protocol] = None,
                       speaker_subset: Optional[str] = 'train',
                       speaker_min_duration: Optional[float] = 60,
                       batch_size: Optional[int] = 32,
                       per_epoch: Optional[float] = 30,
                       parallel: Optional[int] = 1):
        super().__init__()

        self.embedding = embedding
        self.precomputed_ = Precomputed(self.embedding, use_memmap=False)
        if self.precomputed_.sliding_window.start != 0:
            msg = (
                f'Embedding sliding window should start at t=0.'
            )
            raise ValueError(msg)

        self.conversation_protocol = conversation_protocol
        self.conversation_subset = conversation_subset
        self.conversation_duration = conversation_duration

        if speaker_protocol is None:
            speaker_protocol = conversation_protocol
            speaker_subset = conversation_subset
        self.speaker_protocol = speaker_protocol
        self.speaker_subset = speaker_subset
        self.speaker_min_duration = speaker_min_duration

        self._load_metadata(self.speaker_protocol, subset=self.speaker_subset)
        self.speaker_min_duration = speaker_min_duration

        self.conversations_ = ConversationGenerator(
            self.conversation_protocol,
            subset=self.conversation_subset,
            duration=self.conversation_duration,
            window=self.precomputed_.sliding_window)

        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.parallel = parallel

    @property
    def batches_per_epoch(self):

        # duration per epoch
        duration_per_epoch = self.per_epoch * 24 * 60 * 60

        # duration per batch
        duration_per_batch = self.batch_size * self.conversation_duration

        # number of batches per epoch
        return int(np.ceil(duration_per_epoch / duration_per_batch))


    def _load_metadata(self, protocol, subset='train'):

        self.data_ = {}
        self.embedding_ = {}

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # pre-load embeddings
            self.embedding_[current_file['uri']] = \
                self.precomputed_(current_file)

            # get annotation for current file
            annotation = current_file['annotation']

            # loop on each label in current file
            for label in annotation.labels():

                # get all segments with current label
                timeline = annotation.label_timeline(label)

                # store all these in data_ dictionary
                # datum = (segment_generator, duration, current_file, features)
                datum = (timeline, timeline.duration(), current_file)
                self.data_.setdefault(label, []).append(datum)

        # remove labels with less than 'label_min_duration' of speech
        # otherwise those may generate the same segments over and over again
        dropped_labels = set()
        for label, data in self.data_.items():
            total_duration = sum(datum[1] for datum in data)
            if total_duration < self.speaker_min_duration:
                dropped_labels.add(label)

        for label in dropped_labels:
            self.data_.pop(label)

        self.num_speakers_ = len(self.data_)

    def generator(self):

        labels = list(self.data_)
        conversations = self.conversations_()

        while True:

            # generate one random conversation
            y = next(conversations)

            # number of embeddings per speaker
            per_speaker = Counter(y)

            # number of speakers in current conversation
            num_speakers = len(per_speaker)

            # choose 'num_speakers' at random
            speakers = np.random.choice(self.num_speakers_,
                                        size=num_speakers,
                                        replace=False,
                                        p=None)

            per_speaker_X = []
            for i, (target_n, s), in enumerate(zip(per_speaker.values(), speakers)):

                # target_n is number of embedding for ith speaker
                # s is index of random ith speaker

                # load data for this speaker
                # segment_generators, durations, files, features = \
                #     zip(*self.data_[label])
                timelines, durations, files = zip(*self.data_[labels[s]])
                probabilities = np.array(durations) / sum(durations)

                n = 0
                n_left = target_n
                X_i = []
                while n_left > 0:

                    # choose one file at random
                    chosen = np.random.choice(len(files), p=probabilities)

                    # choose one segment at random
                    segment = next(random_segment(timelines[chosen], weighted=False))

                    # get corresponding embedding (and their number)
                    x = self.embedding_[files[chosen]['uri']].crop(
                        segment, mode='center')
                    x_n = len(x)

                    # if we got more than needed, use the first n_left
                    if x_n > n_left:
                        x = x[:n_left]
                        x_n = n_left

                    # append these embeddings and decrement n_left
                    X_i.append(x)
                    n_left -= x_n

                # stack all embeddings for ith speaker
                per_speaker_X.append(np.vstack(X_i))

            X = np.zeros((len(y), per_speaker_X[0].shape[1]))

            for i in range(num_speakers):
                X[np.where(y == i)] = per_speaker_X[i]

            yield {'X': X, 'y': y}

    @property
    def signature(self):
        return {
            'X': {'@': (None, np.stack)},
            'y': {'@': (None, np.stack)},
        }

    @property
    def specifications(self):
        return {
            'X': {'dimension': self.precomputed_.dimension},
            'y': {'n_classes': self.conversations_.max_speakers_},
            'task': TASK_MULTI_CLASS_CLASSIFICATION,
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
