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

import os
import errno
import numpy as np
from pyannote.core import Segment
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature
from pyannote.core.util import string_generator


def mkdir_p(path):
    """Create directory and all its parents if they do not exist

    This is the equivalent of Unix 'mkdir -p path'

    Parameter
    ---------
    path : str
        Path to new directory.

    Reference
    ---------
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc

def to_numpy(annotation, features, labels=None):
    """Convert annotation to numpy array

    Parameters
    ----------
    annotation : pyannote.core.Annotation
        Annotation
    features : pyannote.core.SlidingWindowFeature
        Corresponding features.
    labels : list, optional
        Predefined list of labels.  Defaults to using `annotation` labels.

    Returns
    -------
    y : numpy.ndarray
        (N, K) array where y[t, k] > 0 when labels[k] is active at timestep t.
    labels : list
        List of labels.

    See also
    --------
    See `from_numpy` to convert `y` back to a pyannote.core.Annotation instance
    """

    if labels is None:
        labels = annotation.labels()
    indices = {label: i for i, label in enumerate(labels)}

    # number of samples
    N = len(features)
    # number of classes
    K = len(labels)
    # one-hot encoding
    y = np.zeros((N, K), dtype=np.int8)

    sw = features.sliding_window
    for segment, _, label in annotation.itertracks(yield_label=True):
        try:
            k = indices[label]
        except KeyError as e:
            raise

        for i, j in sw.crop(segment, mode='center', return_ranges=True):
            i = max(0, i)
            j = min(N, j)
            y[i:j, k] += 1

    return y, labels


def from_numpy(y, window, labels=None):
    """Convert numpy array to annotation

    Parameters
    ----------
    y : numpy.ndarray
        Binary (N, K) array where y[t, k] == 1 when labels[k] is active
        at timestep t.
    window : pyannote.core.SlidingWindow or pyannote.core.SlidingWindowFeature
        Corresponding sliding window.
    labels : list, optional
        Predefined list of labels.  Defaults to labels generated by
        `pyannote.core.utils.string_generator`.

    Returns
    -------
    annotation : pyannote.core.Annotation

    See also
    --------
    `to_numpy`
    """

    if np.any(np.abs(y) > 1):
        msg = '`y` must be a binary array (i.e. full of zeros and ones).'
        raise ValueError(msg)

    if isinstance(window, SlidingWindowFeature):
        window = window.sliding_window

    N, K = y.shape

    if labels is None:
        labels = string_generator()
        labels = [next(labels) for _ in range(K)]

    annotation = Annotation()

    y_off = np.zeros((1, K), dtype=np.int8)
    y = np.vstack((y_off, y, y_off))
    diff = np.diff(y, axis=0)
    for k, label in enumerate(labels):
        for t in np.where(diff[:, k] != 0)[0]:
            if diff[t, k] > 0:
                onset_t = window[t].middle
            else:
                segment = Segment(onset_t, window[t - 1].middle)
                annotation[segment, k] = label

    return annotation


