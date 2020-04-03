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

"""
# Signal processing
"""


import numpy as np
import scipy.signal
from pyannote.core import Segment, Timeline
from pyannote.core.utils.generators import pairwise
from sklearn.mixture import GaussianMixture
from pyannote.core.utils.numpy import one_hot_decoding


class Peak(object):
    """Peak detection

    Parameters
    ----------
    alpha : float, optional
        Adaptative threshold coefficient. Defaults to 0.5
    scale : {'absolute', 'relative', 'percentile'}
        Set to 'relative' to make onset/offset relative to min/max.
        Set to 'percentile' to make them relative 1% and 99% percentiles.
        Defaults to 'absolute'.
    min_duration : float, optional
        Defaults to 1 second.
    log_scale : bool, optional
        Set to True to indicate that binarized scores are log scaled.
        Defaults to False.

    """

    def __init__(self, alpha=0.5, min_duration=1.0, scale="absolute", log_scale=False):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.min_duration = min_duration
        self.log_scale = log_scale

    def apply(self, predictions, dimension=0):
        """Peak detection

        Parameter
        ---------
        predictions : SlidingWindowFeature
            Predictions returned by segmentation approaches.

        Returns
        -------
        segmentation : Timeline
            Partition.
        """

        if len(predictions.data.shape) == 1:
            y = predictions.data
        elif predictions.data.shape[1] == 1:
            y = predictions.data[:, 0]
        else:
            y = predictions.data[:, dimension]

        if self.log_scale:
            y = np.exp(y)

        sw = predictions.sliding_window

        precision = sw.step
        order = max(1, int(np.rint(self.min_duration / precision)))
        indices = scipy.signal.argrelmax(y, order=order)[0]

        if self.scale == "absolute":
            mini = 0
            maxi = 1

        elif self.scale == "relative":
            mini = np.nanmin(y)
            maxi = np.nanmax(y)

        elif self.scale == "percentile":
            mini = np.nanpercentile(y, 1)
            maxi = np.nanpercentile(y, 99)

        threshold = mini + self.alpha * (maxi - mini)

        peak_time = np.array([sw[i].middle for i in indices if y[i] > threshold])

        n_windows = len(y)
        start_time = sw[0].start
        end_time = sw[n_windows].end

        boundaries = np.hstack([[start_time], peak_time, [end_time]])
        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation


class Binarize(object):
    """Binarize predictions using onset/offset thresholding

    Parameters
    ----------
    onset : float, optional
        Relative onset threshold. Defaults to 0.5.
    offset : float, optional
        Relative offset threshold. Defaults to 0.5.
    scale : {'absolute', 'relative', 'percentile'}
        Set to 'relative' to make onset/offset relative to min/max.
        Set to 'percentile' to make them relative 1% and 99% percentiles.
        Defaults to 'absolute'.
    log_scale : bool, optional
        Set to True to indicate that binarized scores are log scaled.
        Will apply exponential first. Defaults to False.

    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.
    """

    def __init__(
        self,
        onset=0.5,
        offset=0.5,
        scale="absolute",
        log_scale=False,
        pad_onset=0.0,
        pad_offset=0.0,
        min_duration_on=0.0,
        min_duration_off=0.0,
    ):

        super(Binarize, self).__init__()

        self.onset = onset
        self.offset = offset
        self.scale = scale
        self.log_scale = log_scale

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

    def apply(self, predictions, dimension=0):
        """
        Parameters
        ----------
        predictions : SlidingWindowFeature
            Must be mono-dimensional
        dimension : int, optional
            Which dimension to process
        """

        if len(predictions.data.shape) == 1:
            data = predictions.data
        elif predictions.data.shape[1] == 1:
            data = predictions.data[:, 0]
        else:
            data = predictions.data[:, dimension]

        if self.log_scale:
            data = np.exp(data)

        n_samples = predictions.getNumber()
        window = predictions.sliding_window
        timestamps = [window[i].middle for i in range(n_samples)]

        # initial state
        start = timestamps[0]
        label = data[0] > self.onset

        if self.scale == "absolute":
            mini = 0
            maxi = 1

        elif self.scale == "relative":
            mini = np.nanmin(data)
            maxi = np.nanmax(data)

        elif self.scale == "percentile":
            mini = np.nanpercentile(data, 1)
            maxi = np.nanpercentile(data, 99)

        onset = mini + self.onset * (maxi - mini)
        offset = mini + self.offset * (maxi - mini)

        # timeline meant to store 'active' segments
        active = Timeline()

        for t, y in zip(timestamps[1:], data[1:]):

            # currently active
            if label:
                # switching from active to inactive
                if y < offset:
                    segment = Segment(start - self.pad_onset, t + self.pad_offset)
                    active.add(segment)
                    start = t
                    label = False

            # currently inactive
            else:
                # switching from inactive to active
                if y > onset:
                    start = t
                    label = True

        # if active at the end, add final segment
        if label:
            segment = Segment(start - self.pad_onset, t + self.pad_offset)
            active.add(segment)

        # because of padding, some 'active' segments might be overlapping
        # therefore, we merge those overlapping segments
        active = active.support()

        # remove short 'active' segments
        active = Timeline([s for s in active if s.duration > self.min_duration_on])

        # fill short 'inactive' segments
        inactive = active.gaps()
        for s in inactive:
            if s.duration < self.min_duration_off:
                active.add(s)
        active = active.support()

        return active


class GMMResegmentation(object):
    """
    Parameters
    ----------
    n_components : int, optional
        Number of Gaussian components of the GMMs. Defaults to 128.
    n_iter : int, optional
        Number of EM iterations to train the models. Defaults to 10.
    window : float, optional
        Duration of the smoothing window. Defaults to 1 second.

    Note
    ----
    Recommended feature extraction is to use LibrosaMFCC with 19 static MFCCs
    (no energy nor zero-coefficient)
    >>> feature_extraction = LibrosaMFCC(duration=0.025, step=0.01,
                                         e=False, De=False, DDe=False,
                                         coefs=19, D=False, DD=False,
                                         fmin=0.0, fmax=None, n_mels=40)

    TODO: add option to also resegment speech/non-speech

    """

    def __init__(self, n_components=128, n_iter=10, window=1.0):
        super().__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.window = window

    def apply(self, annotation, features):
        """

        Parameters
        ----------
        annotation : `pyannote.core.Annotation`
            Original annotation to be resegmented.
        features : `SlidingWindowFeature`
            Features

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Resegmented annotation.

        """

        sliding_window = features.sliding_window
        window = np.ones((1, sliding_window.samples(self.window)))

        log_probs = []
        labels = annotation.labels()

        # FIXME: embarrasingly parallel
        for label in labels:

            # gather all features for current label
            span = annotation.label_timeline(label)
            data = features.crop(span, mode="center")

            # train a GMM
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                tol=0.001,
                reg_covar=1e-06,
                max_iter=self.n_iter,
                n_init=1,
                init_params="kmeans",
                weights_init=None,
                means_init=None,
                precisions_init=None,
                random_state=None,
                warm_start=False,
                verbose=0,
                verbose_interval=10,
            ).fit(data)

            # compute log-probability across the whole file
            log_prob = gmm.score_samples(features.data)
            log_probs.append(log_prob)

        # smooth log-probability using a sliding window
        log_probs = scipy.signal.convolve(np.vstack(log_probs), window, mode="same")

        # assign each frame to the most likely label
        y = np.argmax(log_probs, axis=0)

        # reconstruct the annotation
        hypothesis = one_hot_decoding(y, sliding_window, labels=labels)

        # remove original non-speech regions
        return hypothesis.crop(annotation.get_timeline().support())
