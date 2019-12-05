#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

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
from cachetools import LRUCache
CACHE_MAXSIZE = 12

import torch
import torch.nn as nn
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments
from pyannote.database import get_unique_identifier
from pyannote.audio.features import Precomputed
from pyannote.audio.features import RawAudio
from pyannote.audio.train.model import RESOLUTION_FRAME, RESOLUTION_CHUNK


class SequenceLabeling(FileBasedBatchGenerator):
    """Sequence labeling

    Parameters
    ----------
    model : `nn.Module` or `str`
        Model (or path to model). When a path, the directory structure created
        by pyannote command line tools (e.g. pyannote-speech-detection) should
        be kept unchanged so that one can find the corresponding configuration
        file automatically.
    return_intermediate : `int`, optional
        Index of intermediate layer. Returns intermediate hidden state.
        Defaults to returning the final output.
    feature_extraction : callable, optional
        Feature extractor. When not provided and `model` is a path, it is
        inferred directly from the configuration file.
    duration : float, optional
        Subsequence duration, in seconds. When `model` is a path and `duration`
        is not provided, it is inferred directly from the configuration file.
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    batch_size : int, optional
        Defaults to 32.
    device : torch.device, optional
        Defaults to CPU.
    """

    def __init__(self, model=None, feature_extraction=None, duration=1,
                 min_duration=None, step=None, batch_size=32, device=None,
                 return_intermediate=None):

        if not isinstance(model, nn.Module):

            from pyannote.audio.applications.base_labeling import BaseLabeling
            app = BaseLabeling.from_model_pt(model, training=False)

            model = app.model_
            if feature_extraction is None:
                feature_extraction = app.feature_extraction_

            if duration is None:
                duration = app.task_.duration

        self.device = torch.device('cpu') if device is None \
                                          else torch.device(device)
        self.model = model.eval().to(self.device)

        if feature_extraction.augmentation is not None:
            msg = (
                'Data augmentation should not be used '
                'when applying a pre-trained model.'
            )
            raise ValueError(msg)
        self.feature_extraction = feature_extraction

        self.duration = duration
        self.min_duration = min_duration

        generator = SlidingSegments(duration=duration, step=step,
                                    min_duration=min_duration, source='audio')
        self.step = generator.step if step is None else step

        self.resolution_ = self.model.resolution

        # model returns one vector per input frame
        if self.resolution_ == RESOLUTION_FRAME:
            self.resolution_ = self.feature_extraction.sliding_window

        # model returns one vector per input window
        if self.resolution_ == RESOLUTION_CHUNK:
            self.resolution_ = SlidingWindow(duration=self.duration,
                                             step=self.step)

        self.return_intermediate = return_intermediate

        super(SequenceLabeling, self).__init__(
            generator, {'@': (self._process, self.forward)},
            batch_size=batch_size, incomplete=False)

    @property
    def dimension(self):
        return len(self.model.classes)

    @property
    def sliding_window(self):
        if self.return_intermediate is not None:
            return SlidingWindow(duration=self.duration, step=self.step)

        return self.resolution_

    def preprocess(self, current_file):
        """On-demand feature extraction

        Parameters
        ----------
        current_file : dict
            Generated by a pyannote.database.Protocol

        Returns
        -------
        current_file : dict
            Current file with additional "features" entry

        Notes
        -----
        Does nothing when self.feature_extraction is a
        pyannote.audio.features.Precomputed instance.
        """

        # if "features" are precomputed on disk, do nothing
        # as "process_segment" will load just the part we need
        if isinstance(self.feature_extraction, (Precomputed, RawAudio)):
            return current_file

        # if (by chance) current_file already contains "features"
        # do nothing.
        if 'features' in current_file:
            return current_file

        # if we get there, it means that we need to extract features
        # for current_file. let's create a cache to store them...
        if not hasattr(self, 'preprocessed_'):
            self.preprocessed_ = LRUCache(maxsize=CACHE_MAXSIZE)

        # this is the key that will be used to know if "features"
        # already exist in cache
        uri = get_unique_identifier(current_file)

        # if "features" are not cached for current file
        # compute and cache them...
        if uri not in self.preprocessed_:
            features = self.feature_extraction(current_file)
            self.preprocessed_[uri] = features

        # create copy of current_file to prevent "features"
        # from consuming increasing memory...
        preprocessed = dict(current_file)

        # add "features" key
        preprocessed['features'] = self.preprocessed_[uri]

        return preprocessed

    def _process(self, segment, current_file=None):
        """Extract features for current segment

        Parameters
        ----------
        segment : pyannote.core.Segment
        current_file : dict
            Generated by a pyannote.database.Protocol
        """

        # use in-memory "features" whenever they are available
        if 'features' in current_file:
            features = current_file['features']
            return features.crop(segment, mode='center', fixed=self.duration)

        # this line will only happen when self.feature_extraction is a
        # pyannote.audio.features.{Precomputed | RawAudio} instance
        return self.feature_extraction.crop(current_file, segment,
                                            mode='center', fixed=self.duration)

    def forward(self, X):
        """Process (variable-length) sequences

        Parameters
        ----------
        X : `list`
            List of input sequences

        Returns
        -------
        fX : `numpy.ndarray`
            Batch of sequence embeddings.
        """

        lengths = [len(x) for x in X]
        variable_lengths = len(set(lengths)) > 1

        if variable_lengths:
            _, sort = torch.sort(torch.tensor(lengths), descending=True)
            _, unsort = torch.sort(sort)
            sequences = [torch.tensor(X[i],
                                      dtype=torch.float32,
                                      device=self.device) for i in sort]
            packed = pack_sequence(sequences)
        else:
            packed = torch.tensor(np.stack(X),
                                  dtype=torch.float32,
                                  device=self.device)

        if self.return_intermediate is None:
            fX = self.model(packed)
        else:
            _, fX = self.model(packed,
                               return_intermediate=self.return_intermediate)

        fX = fX.detach().to('cpu').numpy()

        if variable_lengths:
            return fX[unsort]

        return fX

    def __call__(self, current_file):
        """Compute predictions on a sliding window

        Parameters
        ----------
        current_file : `dict`
            File (from pyannote.database protocol)

        Returns
        -------
        predictions : `SlidingWindowFeature`
            Predictions.
        """

        # FIXME: make sure this is coherent with other changes related to
        # resolution_...

        # frame and sub-sequence sliding windows
        batches = [batch for batch in self.from_file(current_file,
                                                     incomplete=True)]
        if not batches:
            data = np.zeros((0, self.dimension), dtype=np.float32)
            return SlidingWindowFeature(data, self.resolution_)

        fX = np.vstack(batches)
        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        # this happens for tasks that expects just one label per sequence
        # (rather than one label per frame) or when requesting
        # intermediate representation
        if (fX.ndim == 2) or (self.return_intermediate is not None):
            return SlidingWindowFeature(fX, subsequences)

        # get total number of frames (based on last window end time)
        n_subsequences = len(fX)
        n_frames = self.resolution_.samples(subsequences[n_subsequences].end,
                                   mode='center')

        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, self.dimension), dtype=np.float32)

        # k[i] is the number of sequences that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for subsequence, fX_ in zip(subsequences, fX):

            # indices of frames overlapped by subsequence
            indices = self.resolution_.crop(subsequence,
                                            mode=self.model.alignment,
                                            fixed=self.duration)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        return SlidingWindowFeature(data, self.resolution_)
