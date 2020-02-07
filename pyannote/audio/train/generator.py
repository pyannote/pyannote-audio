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

"""
TODO
"""


import warnings
import threading
import collections
import queue
import time

from abc import ABCMeta, abstractmethod
from typing import Iterator
from typing import Callable

# TODO: move this to pyannote.database
from typing_extensions import Literal
Subset = Literal['train', 'development', 'test']

import numpy as np
from pyannote.audio.features.base import FeatureExtraction
from pyannote.database.protocol.protocol import Protocol

import pescador


class BatchGenerator(metaclass=ABCMeta):
    """Batch generator base class

    Parameters
    ----------
    feature_extraction : `FeatureExtraction`
    protocol : `Protocol`
        pyannote.database protocol used by the generator.
    subset : {'train', 'development', 'test'}, optional
        Subset used by the generator. Defaults to 'train'.
    """

    @abstractmethod
    def __init__(self,
                 feature_extraction : FeatureExtraction,
                 protocol : Protocol,
                 subset : Subset = 'train',
                 **kwargs):
        pass

    @property
    @abstractmethod
    def specifications(self) -> dict:
        """Generator specifications

        Returns
        -------
        specifications : `dict`
            Dictionary describing generator specifications.
        """
        pass

    @property
    @abstractmethod
    def batches_per_epoch(self) -> int:
        """Number of batches per epoch

        Returns
        -------
        n_batches : `int`
            Number of batches to make an epoch.
        """
        pass

    @abstractmethod
    def samples(self) -> Iterator:
        pass

    def __call__(self) -> Iterator:
        batches = pescador.maps.buffer_stream(self.samples(),
                                              self.batch_size,
                                              partial=False,
                                              axis=None)
        while True:
            yield next(batches)



class Background(threading.Thread):
    """Transform a callable into a background iterator.

    Parameters
    ----------
    generator: callable
        Must return an iterator
    prefetch: int, optional
        Maximum number of items that can be prefetched.
        Defaults to 10.

    Usage
    -----
    >>>
    >>> for item in Background(generator, prefetch=10):
    ...     do_something(item)

    """

    def __init__(self, generator: Callable[[], Iterator],
                       prefetch: int = 10):
        super(Background, self).__init__(daemon=True)
        self.generator = generator
        self.prefetch = prefetch

        self.activated_ = True
        self.items_ = generator()

        self.production_time_ = \
            collections.deque([], max(10, 2 * self.prefetch))
        self.consumption_time_ = \
            collections.deque([], max(10, 2 * self.prefetch))
        self.last_ready_ = None

        self.queue_ = queue.Queue(self.prefetch)
        self.start()

    def reset(self) -> None:
        self.production_time_.clear()
        self.consumption_time_.clear()

    def deactivate(self) -> None:
        self.activated_ = False
        # unlock queue stuck at line queue.put() in self.run()
        _ = self.queue_.get()

    @property
    def production_time(self) -> float:
        if len(self.production_time_) < max(10, 2 * self.prefetch):
            return np.NAN
        return np.median(self.production_time_)

    @property
    def consumption_time(self) -> float:
        if len(self.consumption_time_) < max(10, 2 * self.prefetch):
            return np.NAN
        return np.median(self.consumption_time_)

    def run(self) -> None:
        while self.activated_:
            # ask for next item and measure how long it takes to get it
            _t = time.time()
            try:
                item = next(self.items_)
            except StopIteration:
                item = None
            self.production_time_.append(time.time() - _t)
            self.queue_.put(item)

    def __next__(self):
        t = time.time()
        if self.last_ready_ is not None:
            self.consumption_time_.append(t - self.last_ready_)
        next_item = self.queue_.get()
        if next_item is None:
            raise StopIteration
        self.last_ready_ = time.time()
        return next_item

    def __iter__(self):
        return self


class SmartBackground:
    """

    Parameters
    ----------
    generator :
    n_jobs : int, optional
        Maximum number of background jobs.
        Defaults to 4.
    prefetch : int, optional
        Maximum number of iterations that
        each background jobs can prefetch.
        Defaults to 10.
    """

    def __init__(self, generator: Callable[[], Iterator],
                       n_jobs: int = 4,
                       prefetch: int = 10,
                       verbose: bool = False):

        self.generator = generator
        self.n_jobs = n_jobs
        self.prefetch = prefetch
        self.verbose = verbose

        self.generators_ = []
        self._one_more_job()

    def deactivate(self):
        n_jobs = len(self.generators_)
        for _ in range(n_jobs):
            self._one_less_job()

    def _one_more_job(self) -> None:
        """Add one more job to the pool of parallel jobs"""

        n_jobs = len(self.generators_)
        if self.verbose and n_jobs > 0:
            ratio = self.production_time / self.consumption_time
            msg = (
                f'Adding one more job because consumer is '
                f'{ratio:.1f}x faster than current {n_jobs:d} producer(s).'
            )
            warnings.warn(msg)

        self.generators_.append(
            Background(self.generator, prefetch=self.prefetch))

        for g in self.generators_:
            g.reset()

    def _one_less_job(self) -> None:
        """Remove one job from the pool of parallel jobs"""

        n_jobs = len(self.generators_)
        if self.verbose:
            ratio = self.consumption_time / self.production_time
            msg = (
                f'Removing one job because consumer is '
                f'{ratio:.1f}x slower than current {n_jobs:d} producer(s).'
            )
            warnings.warn(msg)

        g = self.generators_.pop()
        g.deactivate()

        for g in self.generators_:
            g.reset()

    @property
    def consumption_time(self):
        return np.mean([g.consumption_time
                        for g in self.generators_])

    @property
    def production_time(self):
        return np.mean([g.production_time
                        for g in self.generators_])

    def __iter__(self):

        while True:

            for g in self.generators_:
                yield next(g)

            consumption_time = self.consumption_time
            production_time = self.production_time

            if np.isnan(consumption_time) or np.isnan(production_time):
                continue

            n_jobs = len(self.generators_)
            ratio = production_time / consumption_time

            # consumption_time < production_time
            if ratio > 1:
                if n_jobs < self.n_jobs:
                    self._one_more_job()

                elif self.verbose:
                    msg = (
                        f'Consumer is {ratio:.1f}x faster than the pool of '
                        f'{n_jobs:d} parallel producer(s) but the maximum '
                        f'number of producers has already been reached.'
                    )
                    warnings.warn(msg)

            # production_time < consumption_time * (n_jobs - 1) / n_jobs
            elif ratio < (n_jobs - 1) / n_jobs:
                self._one_less_job()
