#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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

import functools
import itertools
import numpy as np
import scipy.signal
from multiprocessing import cpu_count
from multiprocessing import Pool

from pyannote.core import Segment, Timeline, Annotation
from pyannote.core.util import pairwise

from pyannote.database.util import get_annotated


def helper_peak_tune(item_prediction, peak=None, metric=None):
    """Apply peak detection on prediction and evaluate the result

    Parameters
    ----------
    item : dict
        Protocol item.
    prediction : SlidingWindowFeature
    peak : Peak, optional
    metric : DiarizationPurityCoverageFMeasure, optional

    Returns
    -------
    fscore : float
    """

    current_file, predictions = item_prediction

    reference = current_file['annotation']

    hypothesis = peak.apply(predictions).to_annotation()
    # remove (reference) non-speech regions
    hypothesis = hypothesis.crop(reference.get_timeline().support(),
                                 mode='intersection')

    uem = get_annotated(current_file)

    # TODO this might take a very long time when hypothesis contains
    # a large number of segments -- find a way to bypass this call
    # and make sure the internal components are accumulated accordingly
    return metric(reference, hypothesis, uem=uem)


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
    def __init__(self, alpha=0.5, min_duration=1.0, scale='absolute',
                 log_scale=False):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.min_duration = min_duration
        self.log_scale = log_scale

    @classmethod
    def tune(cls, items, get_prediction, purity=0.95,
             n_calls=20, n_random_starts=10, n_jobs=-1):
        """Find best set of hyper-parameters using skopt

        Parameters
        ----------
        items : iterable
            Protocol items used as development set. Typically, one would use
            items = protocol.development()
        get_prediction : callable
            Callable that takes an item as input, and returns a prediction as
            a SlidingWindowFeature instances.
        purity : float, optional
            Target purity. Defaults to 0.95.
        n_calls : int, optional
            Number of trials for hyper-parameter optimization. Defaults to 20.
        n_random_starts : int, optional
            Number of trials with random initialization before being smart.
            Defaults to 10.
        n_jobs : int, optional
            Number of parallel job to use. Set to 1 to not use multithreading.
            Defaults to whichever is minimum between number of CPUs and number
            of items.

        Returns
        -------
        best_parameters : dict
            Best set of parameters.
        best_coverage : float
            Best coverage
        """

        import skopt
        import skopt.space

        from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure

        # make sure items can be iterated over and over again
        items = list(items)

        # defaults to whichever is minimum between
        # number of CPUs and number of items
        if n_jobs < 0:
            n_jobs = min(cpu_count(), len(items))

        if n_jobs > 1:
            pool = Pool(n_jobs)

        # compute predictions once and for all
        # NOTE could multithreading speed things up? this is unclear as
        # get_prediction is probably already multithreaded, or (better) even
        # precomputed
        predictions = [get_prediction(item) for item in items]

        def objective_function(params):
            alpha, min_duration, = params
            peak = cls(alpha=alpha, min_duration=min_duration)

            metric = DiarizationPurityCoverageFMeasure()
            process_one_file = functools.partial(
                helper_peak_tune, peak=peak, metric=metric)

            if n_jobs > 1:
                results = list(pool.map(process_one_file,
                                        zip(items, predictions)))
            else:
                results = [process_one_file(item_prediction)
                           for item_prediction in zip(items, predictions)]

            p, c, f = metric.compute_metrics()

            if p < purity:
                return 1.
            else:
                return 1. - c

        # 0 < alpha < 1 || 0 < min_duration < 5s
        space = [skopt.space.Real(0., 1., prior='uniform'),
                 skopt.space.Real(0., 5., prior='uniform')]

        res = skopt.gp_minimize(
            objective_function, space, x0=[[0.5, 2.]],
            n_calls=n_calls, n_random_starts=n_random_starts,
            random_state=1337, verbose=False)

        if n_jobs > 1:
            pool.terminate()

        best_coverage = 1. - res.fun
        return {'alpha': res.x[0], 'min_duration': res.x[1]}, best_coverage


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

        if self.scale == 'absolute':
            mini = 0
            maxi = 1

        elif self.scale == 'relative':
            mini = np.nanmin(data)
            maxi = np.nanmax(data)

        elif self.scale == 'percentile':
            mini = np.nanpercentile(data, 1)
            maxi = np.nanpercentile(data, 99)

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


def helper_binarize_tune(item_prediction,
                         binarizer=None, metric=None, **kwargs):
    """Apply binarizer on prediction and evaluate the result

    Parameters
    ----------
    item : dict
        Protocol item.
    prediction : SlidingWindowFeature
    binarizer : Binarize, optional
    metric : BaseMetric, optional
    **kwargs : dict
        Passed to binarizer.apply()

    Returns
    -------
    value : float
        Metric value
    """

    item, prediction = item_prediction

    uem = get_annotated(item)
    reference = item['annotation']
    hypothesis_timeline = binarizer.apply(prediction, **kwargs)

    # convert from timeline to annnotation
    hypothesis = Annotation(uri=hypothesis_timeline.uri)
    for s, segment in enumerate(hypothesis_timeline):
        hypothesis[segment] = s

    result = metric(reference, hypothesis, uem=uem)

    return result


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

    def __init__(self, onset=0.5, offset=0.5, scale='absolute', log_scale=False,
                 pad_onset=0., pad_offset=0., min_duration_on=0.,
                 min_duration_off=0.):

        super(Binarize, self).__init__()

        self.onset = onset
        self.offset = offset
        self.scale = scale
        self.log_scale = log_scale

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off


    @classmethod
    def tune(cls, items, get_prediction, get_metric=None,
             n_calls=20, n_random_starts=10, n_jobs=-1,
             **kwargs):
        """Find best set of hyper-parameters using skopt

        Parameters
        ----------
        items : iterable
            Protocol items used as development set. Typically, one would use
            items = protocol.development()
        get_prediction : callable
            Callable that takes an item as input, and returns a prediction as
            a SlidingWindowFeature instances.
        get_metric : callable, optional
            Callable that takes no input, and returns a fresh evaluation
            metric. Defaults to pyannote.metrics.detection.DetectionErrorRate.
        n_calls : int, optional
            Number of trials for hyper-parameter optimization. Defaults to 20.
        n_random_starts : int, optional
            Number of trials with random initialization before being smart.
            Defaults to 10.
        n_jobs : int, optional
            Number of parallel job to use. Set to 1 to not use multithreading.
            Defaults to whichever is minimum between number of CPUs and number
            of items.
        **kwargs :
            Optional keyword arguments passed to Binarize.apply().

        Returns
        -------
        best_parameters : dict
            Best set of parameters.
        best_metric : float
            Best metric
        """

        import skopt
        import skopt.space

        if get_metric is None:
            from pyannote.metrics.detection import DetectionErrorRate
            get_metric = DetectionErrorRate

        # make sure items can be iterated over and over again
        items = list(items)

        # defaults to whichever is minimum between
        # number of CPUs and number of items
        if n_jobs < 0:
            n_jobs = min(cpu_count(), len(items))

        if n_jobs > 1:
            pool = Pool(n_jobs)

        # compute predictions once and for all
        # NOTE could multithreading speed things up? this is unclear as
        # get_prediction is probably already multithreaded, or (better) even
        # precomputed
        predictions = [get_prediction(item) for item in items]

        def objective_function(params):
            onset, offset, = params
            binarizer = cls(onset=onset, offset=offset)

            metric = get_metric()
            process_one_file = functools.partial(helper_binarize_tune,
                                                 binarizer=binarizer,
                                                 metric=metric,
                                                 **kwargs)

            if n_jobs > 1:
                results = list(pool.map(process_one_file,
                                        zip(items, predictions)))
            else:
                results = [process_one_file(item_prediction)
                           for item_prediction in zip(items, predictions)]

            return abs(metric)

        space = [skopt.space.Real(0., 1., prior='uniform'),
                 skopt.space.Real(0., 1., prior='uniform')]

        res = skopt.gp_minimize(
            objective_function, space,
            x0=[[0.25, 0.25],[0.5, 0.5], [0.75, 0.75]],
            n_calls=n_calls, n_random_starts=n_random_starts,
            random_state=1337, verbose=False)

        if n_jobs > 1:
            pool.terminate()

        return {'onset': res.x[0], 'offset': res.x[1]}, res.fun

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

        if self.scale == 'absolute':
            mini = 0
            maxi = 1

        elif self.scale == 'relative':
            mini = np.nanmin(data)
            maxi = np.nanmax(data)

        elif self.scale == 'percentile':
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
                    segment = Segment(start - self.pad_onset,
                                      t + self.pad_offset)
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
        active = Timeline(
            [s for s in active if s.duration > self.min_duration_on])

        # fill short 'inactive' segments
        inactive = active.gaps()
        for s in inactive:
            if s.duration < self.min_duration_off:
                active.add(s)
        active = active.support()

        return active
